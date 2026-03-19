from __future__ import annotations

import argparse
import json
import io
import logging
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fuzzywuzzy import process

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    CUSTOM_ATP_FILE,
    CUSTOM_WTA_FILE,
    DATE_FMT,
    FRESH_DAYS,
    LAST_UPDATE_FILE,
    PROCESSED_DIR,
    RAW_ATP,
    RAW_WTA,
    STALE_DAYS,
    WARNING_DAYS,
)
from src.sqlite_storage import latest_match_date

RAW_REPOS = {
    "atp": RAW_ATP,
    "wta": RAW_WTA,
}

FLASHSCORE_TENNIS_URL = "https://www.flashscore.com/tennis/"
SACKMANN_BRANCHES = ("master", "main")
logger = logging.getLogger(__name__)


@dataclass
class PullResult:
    success: bool
    method: str
    message: str
    rows_added: int = 0


class SackmannYearUnavailableError(Exception):
    def __init__(self, tour: str, year: int, urls: list[str]) -> None:
        self.tour = tour
        self.year = year
        self.urls = urls
        super().__init__(
            f"WARNING: {year} Sackmann data not yet available (HTTP 404) — skipping"
        )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_last_update() -> dict[str, Any]:
    if LAST_UPDATE_FILE.exists():
        text = LAST_UPDATE_FILE.read_text(encoding="utf-8-sig")
        if not text.strip():
            return {}
        return json.loads(text)
    return {}


def save_last_update(payload: dict[str, Any]) -> None:
    _ensure_parent(LAST_UPDATE_FILE)
    LAST_UPDATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _yearly_match_files(raw_repo: Path, tour: str) -> list[Path]:
    prefix = f"{tour}_matches_"
    files = [
        p
        for p in raw_repo.glob(f"{prefix}*.csv")
        if p.name.replace(prefix, "").replace(".csv", "").isdigit()
    ]
    return sorted(files)


def _max_date_from_csv(path: Path) -> date | None:
    try:
        df = pd.read_csv(path, usecols=["tourney_date"], dtype={"tourney_date": "Int64"})
    except Exception:
        return None
    if df.empty:
        return None
    series = pd.to_datetime(df["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce").dropna()
    if series.empty:
        return None
    return series.max().date()


def get_latest_match_date_from_raw(tour: str) -> date | None:
    repo = RAW_REPOS[tour]
    files = _yearly_match_files(repo, tour)
    if not files:
        return None

    # Check recent files first for speed.
    for path in reversed(files[-4:]):
        max_dt = _max_date_from_csv(path)
        if max_dt:
            return max_dt

    for path in reversed(files):
        max_dt = _max_date_from_csv(path)
        if max_dt:
            return max_dt

    return None


def get_latest_match_date_from_processed() -> date | None:
    latest = latest_match_date(fallback_to_csv=False)
    if not latest:
        return None
    parsed = pd.to_datetime(latest, errors="coerce")
    return parsed.date() if pd.notna(parsed) else None


def pull_sackmann_repo(repo_path: Path) -> PullResult:
    if not repo_path.exists():
        return PullResult(False, "git", f"Repo path does not exist: {repo_path}")

    try:
        result = subprocess.run(
            ["git", "pull", "origin", "master"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except FileNotFoundError:
        return PullResult(False, "git", "git executable not found")
    except subprocess.TimeoutExpired:
        return PullResult(False, "git", "git pull timeout")

    if result.returncode == 0:
        msg = (result.stdout or "").strip() or "git pull completed"
        return PullResult(True, "git", msg)

    err = (result.stderr or "").strip() or "git pull failed"
    return PullResult(False, "git", err)


def _candidate_sackmann_urls(tour: str, year: int) -> list[str]:
    return [
        f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/{branch}/{tour}_matches_{year}.csv"
        for branch in SACKMANN_BRANCHES
    ]


def _fetch_from_fallback_source(year: int, tour: str) -> pd.DataFrame:
    """
    TODO: wire to a real recent-match fallback source.

    This exists so the updater can keep a clean extension point when Sackmann
    yearly CSVs have not been published yet.
    """
    logger.debug(
        "Fallback source hook not yet implemented for %s %s; returning empty frame",
        tour.upper(),
        year,
    )
    return pd.DataFrame()


def _download_year_csv(tour: str, year: int) -> tuple[pd.DataFrame, str]:
    urls = _candidate_sackmann_urls(tour, year)
    missing_urls: list[str] = []
    errors: list[str] = []

    for url in urls:
        try:
            response = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue

        if response.status_code == 404:
            missing_urls.append(url)
            continue

        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text)), url

    fallback_df = _fetch_from_fallback_source(year, tour)
    if not fallback_df.empty:
        return fallback_df, f"fallback://{tour}/{year}"

    if missing_urls and len(missing_urls) == len(urls):
        raise SackmannYearUnavailableError(tour=tour, year=year, urls=missing_urls)

    if errors:
        raise RuntimeError("; ".join(errors))

    raise RuntimeError(f"No downloadable CSV found for {tour} {year}")


def http_fallback_update(tour: str, dry_run: bool = False) -> PullResult:
    raw_repo = RAW_REPOS[tour]
    raw_repo.mkdir(parents=True, exist_ok=True)
    years = [datetime.now(UTC).year - i for i in range(0, 3)]

    last_error = "No downloadable year file found"
    skipped_years: list[int] = []
    for year in years:
        try:
            remote_df, used_url = _download_year_csv(tour, year)
        except SackmannYearUnavailableError as exc:  # pragma: no cover - network variability
            logger.debug(str(exc))
            skipped_years.append(year)
            last_error = str(exc)
            continue
        except Exception as exc:  # pragma: no cover - network variability
            last_error = f"{year}: {exc}"
            continue

        local_file = raw_repo / f"{tour}_matches_{year}.csv"
        local_rows = 0
        if local_file.exists():
            try:
                local_rows = len(pd.read_csv(local_file))
            except Exception:
                local_rows = 0

        action_prefix = "Would update" if dry_run else "Updated"
        if len(remote_df) > local_rows or not local_file.exists():
            if not dry_run:
                remote_df.to_csv(local_file, index=False)
            return PullResult(
                True,
                "http_probe" if dry_run else "http",
                f"{action_prefix} {local_file.name} via HTTP fallback ({used_url})",
                rows_added=max(0, len(remote_df) - local_rows),
            )

        return PullResult(
            True,
            "http_probe" if dry_run else "http",
            f"No new rows via HTTP fallback ({local_file.name}; source {used_url})",
        )

    skipped_msg = f"; skipped unavailable years: {', '.join(str(y) for y in skipped_years)}" if skipped_years else ""
    return PullResult(
        False,
        "http_probe" if dry_run else "http",
        f"HTTP fallback failed: {last_error}{skipped_msg}",
    )


def download_years_http_fallback(tour: str, years: list[int], dry_run: bool = False) -> dict[str, Any]:
    """Download explicit yearly CSV files from GitHub raw URLs when available."""
    raw_repo = RAW_REPOS[tour]
    raw_repo.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for year in years:
        local_file = raw_repo / f"{tour}_matches_{year}.csv"
        try:
            remote_df, url = _download_year_csv(tour, year)
            if remote_df.empty:
                failed.append(
                    {
                        "year": year,
                        "status": 200,
                        "url": url,
                        "reason": "empty_csv",
                    }
                )
                continue

            old_rows = 0
            if local_file.exists():
                try:
                    old_rows = len(pd.read_csv(local_file, low_memory=False))
                except Exception:
                    old_rows = 0

            if not dry_run:
                remote_df.to_csv(local_file, index=False)
            downloaded.append(
                {
                    "year": year,
                    "rows": int(len(remote_df)),
                    "rows_added": max(0, int(len(remote_df)) - old_rows),
                    "file": str(local_file),
                    "url": url,
                    "dry_run": dry_run,
                }
            )
        except SackmannYearUnavailableError as exc:
            logger.debug(str(exc))
            failed.append(
                {
                    "year": year,
                    "status": 404,
                    "url": exc.urls[0] if exc.urls else None,
                    "reason": "not_yet_available",
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "year": year,
                    "status": None,
                    "url": None,
                    "reason": str(exc),
                }
            )

    return {
        "tour": tour,
        "requested_years": years,
        "downloaded": downloaded,
        "failed": failed,
    }


def _normalize_name(name: str) -> str:
    return " ".join(str(name).lower().replace(".", " ").replace("-", " ").split())


def _build_player_lookup() -> tuple[list[str], dict[str, str], dict[tuple[str, str], list[tuple[str, str]]]]:
    pool: list[str] = []
    name_to_id: dict[str, str] = {}
    last_initial_idx: dict[tuple[str, str], list[tuple[str, str]]] = {}

    player_files = [
        RAW_ATP / "atp_players.csv",
        RAW_WTA / "wta_players.csv",
    ]
    for path in player_files:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["player_id", "name_first", "name_last"], low_memory=False)
        except Exception:
            continue

        for pid, first, last in df.itertuples(index=False):
            full = f"{first} {last}".strip()
            if not full or full.lower() == "nan":
                continue
            pool.append(full)
            try:
                player_id = str(int(float(pid)))
            except Exception:
                player_id = str(pid)
            name_to_id[full] = player_id

            first_n = _normalize_name(first)
            last_n = _normalize_name(last)
            if first_n and last_n:
                key = (last_n, first_n[0])
                last_initial_idx.setdefault(key, []).append((full, player_id))

    return sorted(set(pool)), name_to_id, last_initial_idx


def _load_name_overrides(path: Path = Path("name_overrides.json")) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    data = json.loads(text) if text.strip() else {}
    return {_normalize_name(k): str(v).strip() for k, v in data.items() if isinstance(k, str)}


def _resolve_player(
    raw_name: str,
    player_pool: list[str],
    name_to_id: dict[str, str],
    last_initial_idx: dict[tuple[str, str], list[tuple[str, str]]],
    overrides: dict[str, str],
) -> tuple[str, str | None, int]:
    norm = _normalize_name(raw_name)
    if norm in overrides:
        overridden = overrides[norm]
        return overridden, name_to_id.get(overridden), 100

    m = re.match(r"^(.+?)\s+([A-Za-z])\.?$", str(raw_name).strip())
    if m:
        key = (_normalize_name(m.group(1)), m.group(2).lower())
        candidates = last_initial_idx.get(key, [])
        if len(candidates) == 1:
            c_name, c_id = candidates[0]
            return c_name, c_id, 100
        if candidates:
            best = process.extractOne(raw_name, [x[0] for x in candidates])
            if best:
                chosen = best[0]
                chosen_id = dict(candidates).get(chosen)
                return chosen, chosen_id, int(best[1])

    if not player_pool:
        return raw_name, None, 0
    best = process.extractOne(raw_name, player_pool)
    if not best:
        return raw_name, None, 0
    chosen, score = str(best[0]), int(best[1])
    if score < 85:
        return raw_name, None, score
    return chosen, name_to_id.get(chosen), score


def _parse_header_for_results(text: str) -> tuple[str | None, str | None, str | None, str | None]:
    # Example: "Indian Wells (USA), hard\nATP - SINGLES: \n1\n2"
    parts = [p.strip() for p in re.split(r"[|\n]+", text) if p.strip()]
    if len(parts) < 2:
        return None, None, None, None

    tournament = parts[0]
    category = parts[1].upper()

    if "ATP - SINGLES" in category:
        tour = "atp"
    elif "WTA - SINGLES" in category:
        tour = "wta"
    elif "CHALLENGER MEN - SINGLES" in category:
        tour = "atp"
    elif "CHALLENGER WOMEN - SINGLES" in category:
        tour = "wta"
    elif "ITF MEN - SINGLES" in category:
        tour = "atp"
    elif "ITF WOMEN - SINGLES" in category:
        tour = "wta"
    else:
        return None, None, None, None

    surface = None
    if "," in tournament:
        surface = tournament.split(",")[-1].strip().title()
        if "Hard" in surface:
            surface = "Hard"
        elif "Clay" in surface:
            surface = "Clay"
        elif "Grass" in surface:
            surface = "Grass"
        elif "Carpet" in surface:
            surface = "Carpet"

    tournament_upper = tournament.upper()
    if any(x in tournament_upper for x in ["WIMBLEDON", "ROLAND GARROS", "AUSTRALIAN OPEN", "US OPEN"]):
        level = "G"
    elif "CHALLENGER" in category or "ITF" in category:
        level = "C"
    else:
        level = "A"

    return tour, tournament, surface, level


def _score_from_row(row: Any) -> tuple[str | None, int]:
    home_parts = row.find_elements("css selector", "div.event__part--home")
    away_parts = row.find_elements("css selector", "div.event__part--away")
    n = min(len(home_parts), len(away_parts))
    if n == 0:
        return None, 0

    sets = []
    for i in range(n):
        h = (home_parts[i].text or "").strip()
        a = (away_parts[i].text or "").strip()
        if h.isdigit() and a.isdigit():
            sets.append(f"{h}-{a}")
    if not sets:
        return None, 0
    return " ".join(sets), len(sets)


def _append_custom_matches(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema_cols = [
        "tourney_id",
        "tourney_name",
        "surface",
        "draw_size",
        "tourney_level",
        "tourney_date",
        "match_num",
        "winner_id",
        "winner_name",
        "loser_id",
        "loser_name",
        "score",
        "best_of",
        "round",
        "source",
    ]
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, low_memory=False)
    else:
        existing = pd.DataFrame(columns=schema_cols)

    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return 0
    for col in schema_cols:
        if col not in incoming.columns:
            incoming[col] = pd.NA
        if col not in existing.columns:
            existing[col] = pd.NA

    before = len(existing)
    merged = pd.concat([existing[schema_cols], incoming[schema_cols]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["tourney_id", "match_num", "winner_id", "loser_id"], keep="last")
    merged.to_csv(path, index=False)
    return max(0, len(merged) - before)


def get_staleness_status(last_match_date: date | None, today: date | None = None) -> dict[str, Any]:
    if today is None:
        today = datetime.now(UTC).date()

    if last_match_date is None:
        return {
            "days_behind": None,
            "status": "unknown",
            "message": "No match data found yet.",
        }

    days_behind = (today - last_match_date).days
    if days_behind <= FRESH_DAYS:
        status = "fresh"
        message = "Data is fresh."
    elif days_behind <= WARNING_DAYS:
        status = "warning"
        message = (
            f"Data is {days_behind} days behind. Sackmann may be delayed. "
            "Predictions are still usable."
        )
    elif days_behind <= STALE_DAYS:
        status = "stale"
        message = (
            f"Data is {days_behind} days stale. "
            "Consider adding recent results manually."
        )
    else:
        status = "critical"
        message = (
            f"Data is critically stale ({days_behind} days). "
            "Model accuracy is likely degrading."
        )

    return {
        "days_behind": days_behind,
        "status": status,
        "message": message,
    }


def update_data_sources(dry_run: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tours": {},
        "dry_run": dry_run,
    }

    for tour, repo_path in RAW_REPOS.items():
        if dry_run:
            git_result = PullResult(True, "dry_run", "Skipped git pull in dry-run mode")
            final_result = http_fallback_update(tour, dry_run=True)
        else:
            git_result = pull_sackmann_repo(repo_path)
            final_result = git_result
            if not git_result.success:
                final_result = http_fallback_update(tour)

        latest_raw = get_latest_match_date_from_raw(tour)
        report["tours"][tour] = {
            "success": final_result.success,
            "method": final_result.method,
            "message": final_result.message,
            "rows_added": final_result.rows_added,
            "latest_raw_match_date": latest_raw.strftime(DATE_FMT) if latest_raw else None,
            "git_success": git_result.success,
        }

    # Collect candidate dates from raw Sackmann files
    candidate_dates: list[date] = [
        datetime.strptime(info["latest_raw_match_date"], DATE_FMT).date()
        for info in report["tours"].values()
        if info.get("latest_raw_match_date")
    ]

    # Also consider the processed master files (which include custom/scraped matches)
    processed_date = get_latest_match_date_from_processed()
    if processed_date is not None:
        candidate_dates.append(processed_date)

    # Preserve the previous last_new_match if it's more recent (e.g. from pipeline runs)
    prev_state = load_last_update()
    prev_last = prev_state.get("last_new_match")
    if isinstance(prev_last, str) and prev_last:
        try:
            candidate_dates.append(datetime.strptime(prev_last, DATE_FMT).date())
        except Exception:
            pass

    last_new_match = max(candidate_dates) if candidate_dates else None

    staleness = get_staleness_status(last_new_match)
    report["staleness"] = staleness

    if not dry_run:
        state = prev_state
        state.update(
            {
                "updated_at": report["timestamp"],
                "sackmann_pull": report["timestamp"],
                "tours": report["tours"],
                "last_new_match": last_new_match.strftime(DATE_FMT) if last_new_match else None,
                "staleness": staleness,
            }
        )
        save_last_update(state)

    return report


def scrape_flashscore_results() -> dict[str, Any]:
    """Scrape finished ATP/WTA singles rows from Flashscore into custom match CSVs."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except Exception as exc:
        return {
            "success": False,
            "message": f"Selenium unavailable: {exc}",
            "rows_added": {"atp": 0, "wta": 0},
        }

    player_pool, name_to_id, last_initial_idx = _build_player_lookup()
    overrides = _load_name_overrides()

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_bin = os.environ.get("CHROME_BIN") or os.environ.get("GOOGLE_CHROME_BIN")
    chromedriver_path = os.environ.get("CHROMEDRIVER_PATH")
    if chrome_bin:
        options.binary_location = chrome_bin

    rows_by_tour: dict[str, list[dict[str, Any]]] = {"atp": [], "wta": []}
    skipped = 0

    service = Service(executable_path=chromedriver_path) if chromedriver_path else None
    driver = webdriver.Chrome(service=service, options=options)
    try:
        try:
            driver.get(FLASHSCORE_TENNIS_URL)
            time.sleep(random.uniform(2.0, 3.0))

            root = driver.find_element("css selector", "div.sportName.tennis")
            children = root.find_elements("xpath", "./*")

            current_tour: str | None = None
            current_tournament: str | None = None
            current_surface: str | None = None
            current_level: str | None = None
            league_targets: list[tuple[str, str, str | None, str | None]] = []

            tourney_date = datetime.now(UTC).strftime("%Y%m%d")

            for child in children:
                classes = child.get_attribute("class") or ""
                if "headerLeague__wrapper" in classes:
                    current_tour, current_tournament, current_surface, current_level = _parse_header_for_results(child.text)
                    if current_tour in {"atp", "wta"}:
                        try:
                            href = child.find_element("css selector", "a.headerLeague__title").get_attribute("href")
                            if href:
                                league_targets.append((current_tour, href, current_tournament, current_surface))
                        except Exception:
                            pass
                    continue

                if "event__match" not in classes:
                    continue
                if current_tour not in {"atp", "wta"}:
                    continue
                # keep finished matches only
                stage = (child.text or "")
                if "Finished" not in stage:
                    continue

                try:
                    home_el = child.find_element("css selector", "div.event__participant--home")
                    away_el = child.find_element("css selector", "div.event__participant--away")
                    home_name_raw = (home_el.text or "").strip()
                    away_name_raw = (away_el.text or "").strip()
                    if not home_name_raw or not away_name_raw:
                        skipped += 1
                        continue

                    score_text, set_count = _score_from_row(child)
                    if not score_text:
                        skipped += 1
                        continue

                    home_full, home_id, _ = _resolve_player(
                        home_name_raw, player_pool, name_to_id, last_initial_idx, overrides
                    )
                    away_full, away_id, _ = _resolve_player(
                        away_name_raw, player_pool, name_to_id, last_initial_idx, overrides
                    )
                    if not home_id or not away_id:
                        skipped += 1
                        continue

                    home_is_winner = "fontExtraBold" in (home_el.get_attribute("class") or "")
                    if home_is_winner:
                        winner_id, winner_name = home_id, home_full
                        loser_id, loser_name = away_id, away_full
                    else:
                        winner_id, winner_name = away_id, away_full
                        loser_id, loser_name = home_id, home_full

                    row_id = child.get_attribute("id") or ""
                    match_id = row_id.split("_")[-1] if "_" in row_id else row_id
                    best_of = 5 if set_count >= 4 else 3

                    match_num = abs(hash(match_id)) % 1_000_000 if match_id else random.randint(1, 999999)
                    tourney_id = f"fs-{current_tour}-{tourney_date}-{match_id or match_num}"

                    rows_by_tour[current_tour].append(
                        {
                            "tourney_id": tourney_id,
                            "tourney_name": current_tournament,
                            "surface": current_surface or "Hard",
                            "draw_size": pd.NA,
                            "tourney_level": current_level or "A",
                            "tourney_date": tourney_date,
                            "match_num": match_num,
                            "winner_id": winner_id,
                            "winner_name": winner_name,
                            "loser_id": loser_id,
                            "loser_name": loser_name,
                            "score": score_text,
                            "best_of": best_of,
                            "round": "R32",
                            "source": "custom",
                        }
                    )
                except Exception:
                    skipped += 1
                    continue

            # Second pass: parse tournament pages (recent completed matches) for richer 2025/2026 coverage.
            seen_targets = set()
            for tour, href, tournament_name, surface_name in league_targets:
                key = (tour, href)
                if key in seen_targets:
                    continue
                seen_targets.add(key)

                try:
                    driver.get(href)
                    time.sleep(random.uniform(2.0, 3.0))

                    # Prefer explicit RESULTS tab if available.
                    for tab in driver.find_elements("css selector", "a.tabs__tab"):
                        if (tab.text or "").strip().upper() == "RESULTS":
                            tab.click()
                            time.sleep(random.uniform(2.0, 3.0))
                            break

                    matches = driver.find_elements("css selector", "div.event__match")
                    for match in matches:
                        cls = match.get_attribute("class") or ""
                        if "event__match--scheduled" in cls:
                            continue

                        try:
                            home_el = match.find_element("css selector", "div.event__participant--home")
                            away_el = match.find_element("css selector", "div.event__participant--away")
                            home_name_raw = (home_el.text or "").strip()
                            away_name_raw = (away_el.text or "").strip()
                            if not home_name_raw or not away_name_raw:
                                skipped += 1
                                continue

                            score_text, set_count = _score_from_row(match)
                            if not score_text:
                                skipped += 1
                                continue

                            time_text = (match.find_element("css selector", "div.event__time").text or "").strip()
                            date_match = re.search(r"(\d{2})\.(\d{2})\.", time_text)
                            if date_match:
                                day = int(date_match.group(1))
                                month = int(date_match.group(2))
                                year = datetime.now(UTC).year
                                dt = datetime(year, month, day, tzinfo=UTC)
                                tourney_date_match = dt.strftime("%Y%m%d")
                            else:
                                tourney_date_match = datetime.now(UTC).strftime("%Y%m%d")

                            home_full, home_id, _ = _resolve_player(
                                home_name_raw, player_pool, name_to_id, last_initial_idx, overrides
                            )
                            away_full, away_id, _ = _resolve_player(
                                away_name_raw, player_pool, name_to_id, last_initial_idx, overrides
                            )
                            if not home_id or not away_id:
                                skipped += 1
                                continue

                            home_is_winner = "fontExtraBold" in (home_el.get_attribute("class") or "")
                            if home_is_winner:
                                winner_id, winner_name = home_id, home_full
                                loser_id, loser_name = away_id, away_full
                            else:
                                winner_id, winner_name = away_id, away_full
                                loser_id, loser_name = home_id, home_full

                            row_id = match.get_attribute("id") or ""
                            match_id = row_id.split("_")[-1] if "_" in row_id else row_id
                            best_of = 5 if set_count >= 4 else 3
                            match_num = abs(hash(match_id or (home_name_raw + away_name_raw + tourney_date_match))) % 1_000_000

                            rows_by_tour[tour].append(
                                {
                                    "tourney_id": f"fs-{tour}-{tourney_date_match}-{match_id or match_num}",
                                    "tourney_name": tournament_name or f"Flashscore {tour.upper()}",
                                    "surface": (surface_name or "Hard"),
                                    "draw_size": pd.NA,
                                    "tourney_level": "A",
                                    "tourney_date": tourney_date_match,
                                    "match_num": match_num,
                                    "winner_id": winner_id,
                                    "winner_name": winner_name,
                                    "loser_id": loser_id,
                                    "loser_name": loser_name,
                                    "score": score_text,
                                    "best_of": best_of,
                                    "round": "R32",
                                    "source": "custom",
                                }
                            )
                        except Exception:
                            skipped += 1
                            continue
                except Exception:
                    skipped += 1
                    continue
        except Exception as exc:
            return {
                "success": False,
                "message": f"Flashscore results scrape failed: {exc}",
                "rows_scraped": {"atp": 0, "wta": 0},
                "rows_added": {"atp": 0, "wta": 0},
                "rows_skipped": skipped,
            }
    finally:
        driver.quit()

    atp_added = _append_custom_matches(CUSTOM_ATP_FILE, rows_by_tour["atp"])
    wta_added = _append_custom_matches(CUSTOM_WTA_FILE, rows_by_tour["wta"])

    return {
        "success": True,
        "message": "Flashscore finished results scraped into custom files.",
        "rows_scraped": {
            "atp": len(rows_by_tour["atp"]),
            "wta": len(rows_by_tour["wta"]),
        },
        "rows_added": {
            "atp": atp_added,
            "wta": wta_added,
        },
        "rows_skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Update local tennis raw data sources")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe remote sources and log what would happen without writing files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    report = update_data_sources(dry_run=args.dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
