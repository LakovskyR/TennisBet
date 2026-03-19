from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fuzzywuzzy import process
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

from config import ODDS_HISTORY_FILE, ODDS_MOVEMENT_FILE, ODDS_UPCOMING_FILE
from src.odds_tracker import write_odds_movement_snapshot
from src.sqlite_storage import initialize_database, sync_odds_frame

FLASHSCORE_TENNIS_URL = "https://www.flashscore.com/tennis/"
FUZZY_THRESHOLD = 85
RATE_LIMIT_MIN = 2.0
RATE_LIMIT_MAX = 3.0

OUTPUT_COLUMNS = [
    "match_date",
    "match_time",
    "tour",
    "tournament",
    "surface",
    "player_1",
    "player_2",
    "player_1_resolved",
    "player_2_resolved",
    "player_1_id",
    "player_2_id",
    "player_1_match_score",
    "player_2_match_score",
    "odds_p1",
    "odds_p2",
    "bookmaker",
    "bookmaker_count",
    "aggregation_method",
    "source_url",
    "match_id",
    "captured_at",
]

logger = logging.getLogger("odds_scraper")


def _setup_logging() -> None:
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


def _sleep_rate_limit() -> None:
    time.sleep(random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX))


def _normalize_name(name: str) -> str:
    return " ".join(str(name).lower().replace(".", " ").replace("-", " ").split())


def _load_name_overrides(path: Path = Path("name_overrides.json")) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    data = json.loads(text) if text.strip() else {}
    return {_normalize_name(k): str(v).strip() for k, v in data.items() if isinstance(k, str)}


def _build_player_pool() -> tuple[list[str], dict[str, str], dict[tuple[str, str], list[tuple[str, str]]]]:
    pool: list[str] = []
    name_to_id: dict[str, str] = {}
    last_initial_index: dict[tuple[str, str], list[tuple[str, str]]] = {}

    player_files = [
        Path("data/raw/tennis_atp/atp_players.csv"),
        Path("data/raw/tennis_wta/wta_players.csv"),
    ]

    for file in player_files:
        if not file.exists():
            continue
        try:
            df = pd.read_csv(file, usecols=["player_id", "name_first", "name_last"], low_memory=False)
        except Exception as exc:
            logger.warning("Failed to read player file %s: %s", file, exc)
            continue

        for pid, first, last in df.itertuples(index=False):
            name = f"{first} {last}".strip()
            if not name or name.lower() == "nan":
                continue
            pool.append(name)
            try:
                player_id = str(int(float(pid)))
            except Exception:
                player_id = str(pid)
            name_to_id[name] = player_id
            first_norm = _normalize_name(first)
            last_norm = _normalize_name(last)
            if first_norm and last_norm:
                key = (last_norm, first_norm[0])
                last_initial_index.setdefault(key, []).append((name, player_id))

    dedup_pool = sorted(set(pool))
    return dedup_pool, name_to_id, last_initial_index


def _resolve_player_name(
    raw_name: str,
    player_pool: list[str],
    name_to_id: dict[str, str],
    last_initial_index: dict[tuple[str, str], list[tuple[str, str]]],
    overrides: dict[str, str],
) -> tuple[str, str | None, int]:
    normalized = _normalize_name(raw_name)

    if normalized in overrides:
        overridden = overrides[normalized]
        return overridden, name_to_id.get(overridden), 100

    # Flashscore commonly uses "Lastname I." format. Resolve deterministically first.
    m_last_first = re.match(r"^(.+?)\s+([A-Za-z])\.?$", str(raw_name).strip())
    m_first_last = re.match(r"^([A-Za-z])\.?\s+(.+)$", str(raw_name).strip())
    key: tuple[str, str] | None = None

    if m_last_first:
        last = _normalize_name(m_last_first.group(1))
        initial = m_last_first.group(2).lower()
        key = (last, initial)
    elif m_first_last:
        initial = m_first_last.group(1).lower()
        last = _normalize_name(m_first_last.group(2))
        key = (last, initial)

    if key and key in last_initial_index:
        candidates = last_initial_index[key]
        if len(candidates) == 1:
            cand_name, cand_id = candidates[0]
            return cand_name, cand_id, 100
        choice = process.extractOne(raw_name, [c[0] for c in candidates])
        if choice:
            chosen_name = choice[0]
            chosen_id = dict(candidates).get(chosen_name)
            return chosen_name, chosen_id, int(choice[1])

    if not player_pool:
        return raw_name, None, 0

    match = process.extractOne(raw_name, player_pool)
    if not match:
        return raw_name, None, 0

    name, score = match[0], int(match[1])
    if score < FUZZY_THRESHOLD:
        return raw_name, None, score
    return name, name_to_id.get(name), score


def _parse_header(text: str) -> tuple[str | None, str | None, str | None]:
    """Return (tour, tournament, surface) from Flashscore header text."""
    if not text:
        return None, None, None

    # Example raw text:
    # "Indian Wells (USA), hard\nATP - SINGLES: \n1\n2"
    parts = [p.strip() for p in re.split(r"[|\n]+", text) if p.strip()]
    if len(parts) < 2:
        return None, None, None

    left, category = parts[0], parts[1].upper()
    if "ATP - SINGLES" in category:
        tour = "atp"
    elif "WTA - SINGLES" in category:
        tour = "wta"
    else:
        return None, None, None

    tournament = left
    surface = None
    if "," in left:
        surface = left.split(",")[-1].strip()

    return tour, tournament, surface


def _init_driver() -> webdriver.Chrome:
    options = Options()
    chrome_bin = os.environ.get("CHROME_BIN") or os.environ.get("GOOGLE_CHROME_BIN")
    chromedriver_path = os.environ.get("CHROMEDRIVER_PATH")

    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    if chrome_bin:
        options.binary_location = chrome_bin

    service = Service(executable_path=chromedriver_path) if chromedriver_path else None
    return webdriver.Chrome(service=service, options=options)


def _click_odds_tab(driver: webdriver.Chrome) -> None:
    tabs = driver.find_elements(By.CSS_SELECTOR, ".filters__tab")
    if not tabs:
        raise TimeoutException("Could not find filter tabs on Flashscore page.")

    clicked = False
    for tab in tabs:
        if (tab.text or "").strip().upper() == "ODDS":
            tab.click()
            clicked = True
            break
    if not clicked:
        raise TimeoutException("ODDS tab not found on Flashscore page.")


def _scrape_match_market_odds(driver: webdriver.Chrome, source_url: str) -> tuple[float | None, float | None, int]:
    odds_url = source_url.split("?", 1)[0].rstrip("/") + "/odds/home-away/full-time/"
    driver.get(odds_url)
    _sleep_rate_limit()

    rows = driver.find_elements(By.CSS_SELECTOR, ".ui-table__body .ui-table__row")
    values: list[tuple[float, float]] = []
    for row in rows:
        try:
            odd_cells = row.find_elements(By.CSS_SELECTOR, ".oddsCell__odd")
            if len(odd_cells) < 2:
                continue
            odd1 = float(odd_cells[0].text.strip())
            odd2 = float(odd_cells[1].text.strip())
            if odd1 > 1.0 and odd2 > 1.0:
                values.append((odd1, odd2))
        except Exception:
            continue

    if not values:
        return None, None, 0

    odds_df = pd.DataFrame(values, columns=["odds_p1", "odds_p2"])
    return (
        round(float(odds_df["odds_p1"].median()), 3),
        round(float(odds_df["odds_p2"].median()), 3),
        int(len(odds_df)),
    )


def scrape_flashscore_upcoming_odds() -> pd.DataFrame:
    _setup_logging()

    player_pool, name_to_id, last_initial_index = _build_player_pool()
    overrides = _load_name_overrides()

    driver = _init_driver()
    rows_out: list[dict[str, Any]] = []

    try:
        logger.info("Opening Flashscore tennis page")
        driver.get(FLASHSCORE_TENNIS_URL)
        _sleep_rate_limit()

        _click_odds_tab(driver)
        _sleep_rate_limit()

        sport_root = driver.find_element(By.CSS_SELECTOR, "div.sportName.tennis")
        children = sport_root.find_elements(By.XPATH, "./*")

        current_tour: str | None = None
        current_tournament: str | None = None
        current_surface: str | None = None

        captured_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        match_date = datetime.now(UTC).strftime("%Y-%m-%d")

        for child in children:
            classes = child.get_attribute("class") or ""

            if "headerLeague__wrapper" in classes:
                tour, tournament, surface = _parse_header(child.text)
                current_tour = tour
                current_tournament = tournament
                current_surface = surface
                continue

            if "event__match" not in classes:
                continue
            if "event__match--scheduled" not in classes:
                continue
            if current_tour not in {"atp", "wta"}:
                continue

            try:
                p1 = child.find_element(By.CSS_SELECTOR, "div.event__participant--home").text.strip()
                p2 = child.find_element(By.CSS_SELECTOR, "div.event__participant--away").text.strip()
                match_time = child.find_element(By.CSS_SELECTOR, "div.event__time").text.strip()

                odd1_text = child.find_element(By.CSS_SELECTOR, "div.event__odd--odd1").text.strip()
                odd2_text = child.find_element(By.CSS_SELECTOR, "div.event__odd--odd2").text.strip()

                odds_p1 = float(odd1_text)
                odds_p2 = float(odd2_text)

                resolved_p1, p1_id, p1_score = _resolve_player_name(
                    p1, player_pool, name_to_id, last_initial_index, overrides
                )
                resolved_p2, p2_id, p2_score = _resolve_player_name(
                    p2, player_pool, name_to_id, last_initial_index, overrides
                )

                link_el = child.find_element(By.CSS_SELECTOR, "a.eventRowLink")
                source_url = link_el.get_attribute("href")

                row_id = child.get_attribute("id") or ""
                match_id = row_id.split("_")[-1] if "_" in row_id else row_id

                rows_out.append(
                    {
                        "match_date": match_date,
                        "match_time": match_time,
                        "tour": current_tour,
                        "tournament": current_tournament,
                        "surface": current_surface,
                        "player_1": p1,
                        "player_2": p2,
                        "player_1_resolved": resolved_p1,
                        "player_2_resolved": resolved_p2,
                        "player_1_id": p1_id,
                        "player_2_id": p2_id,
                        "player_1_match_score": p1_score,
                        "player_2_match_score": p2_score,
                        "odds_p1": odds_p1,
                        "odds_p2": odds_p2,
                        "source_url": source_url,
                        "match_id": match_id,
                        "captured_at": captured_at,
                    }
                )
            except Exception as exc:
                logger.debug("Skipping row parse failure: %s", exc)
                continue

        df = pd.DataFrame(rows_out)
        if not df.empty:
            for row in rows_out:
                source_url = str(row.get("source_url") or "").strip()
                if not source_url:
                    row["bookmaker"] = "flashscore_event_page"
                    row["bookmaker_count"] = 1
                    row["aggregation_method"] = "single_source"
                    continue
                try:
                    median_p1, median_p2, bookmaker_count = _scrape_match_market_odds(driver, source_url)
                except Exception as exc:
                    logger.debug("Market odds scrape failed for %s: %s", source_url, exc)
                    median_p1 = None
                    median_p2 = None
                    bookmaker_count = 0

                if median_p1 is not None and median_p2 is not None and bookmaker_count > 0:
                    row["odds_p1"] = median_p1
                    row["odds_p2"] = median_p2
                    row["bookmaker"] = "flashscore_market_median"
                    row["bookmaker_count"] = bookmaker_count
                    row["aggregation_method"] = "median"
                else:
                    row["bookmaker"] = "flashscore_event_page"
                    row["bookmaker_count"] = 1
                    row["aggregation_method"] = "single_source"

            df = pd.DataFrame(rows_out)
            df = df[OUTPUT_COLUMNS].drop_duplicates(subset=["match_id", "captured_at"], keep="last")
        else:
            df = pd.DataFrame(columns=OUTPUT_COLUMNS)

        logger.info("Scraped %d upcoming ATP/WTA singles rows with odds", len(df))
        return df

    finally:
        driver.quit()


def _ensure_manual_template() -> None:
    ODDS_UPCOMING_FILE.parent.mkdir(parents=True, exist_ok=True)
    initialize_database()
    if not ODDS_UPCOMING_FILE.exists() or ODDS_UPCOMING_FILE.stat().st_size == 0:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(ODDS_UPCOMING_FILE, index=False)


def _write_outputs(upcoming_df: pd.DataFrame) -> tuple[Path, Path]:
    ODDS_UPCOMING_FILE.parent.mkdir(parents=True, exist_ok=True)
    initialize_database()

    # Overwrite today's odds snapshot
    upcoming_df.to_csv(ODDS_UPCOMING_FILE, index=False)

    # Append-only archive
    if ODDS_HISTORY_FILE.exists() and ODDS_HISTORY_FILE.stat().st_size > 0:
        history = pd.read_csv(ODDS_HISTORY_FILE, low_memory=False)
    else:
        history = pd.DataFrame(columns=OUTPUT_COLUMNS)

    for col in OUTPUT_COLUMNS:
        if col not in history.columns:
            history[col] = pd.NA

    if history.empty:
        history = upcoming_df[OUTPUT_COLUMNS].copy()
    else:
        history = pd.concat([history[OUTPUT_COLUMNS], upcoming_df[OUTPUT_COLUMNS]], ignore_index=True)
    history.to_csv(ODDS_HISTORY_FILE, index=False)
    sync_odds_frame(upcoming_df, mark_current_upcoming=False, source_name="odds_history")
    sync_odds_frame(upcoming_df, mark_current_upcoming=True, source_name="odds_upcoming")

    return ODDS_UPCOMING_FILE, ODDS_HISTORY_FILE


def refresh_odds() -> dict[str, Any]:
    _setup_logging()

    try:
        upcoming = scrape_flashscore_upcoming_odds()
        upcoming_path, history_path = _write_outputs(upcoming)
        movement_report = write_odds_movement_snapshot(
            output_path=ODDS_MOVEMENT_FILE,
            current_df=upcoming,
        )
        return {
            "success": True,
            "method": "selenium_flashscore",
            "message": f"Scraped and saved {len(upcoming)} upcoming rows.",
            "upcoming_rows": int(len(upcoming)),
            "upcoming_file": str(upcoming_path),
            "history_file": str(history_path),
            "movement_file": movement_report["output_path"],
            "movement_rows": movement_report["rows"],
            "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except (TimeoutException, WebDriverException, Exception) as exc:
        logger.exception("Flashscore scraping failed; falling back to manual CSV path")
        _ensure_manual_template()
        return {
            "success": False,
            "method": "manual_fallback",
            "message": f"Flashscore scraping failed: {exc}",
            "upcoming_rows": 0,
            "upcoming_file": str(ODDS_UPCOMING_FILE),
            "history_file": str(ODDS_HISTORY_FILE),
            "movement_file": str(ODDS_MOVEMENT_FILE),
            "movement_rows": 0,
            "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


def main() -> None:
    report = refresh_odds()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
