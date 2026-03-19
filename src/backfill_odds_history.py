from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from config import ODDS_HISTORY_FILE, PROCESSED_DIR
from src.backfill_matches import ALL_REGISTRIES
from src.odds_scraper import (
    OUTPUT_COLUMNS,
    _build_player_pool,
    _init_driver,
    _load_name_overrides,
    _normalize_name,
    _resolve_player_name,
    _scrape_match_market_odds,
    _setup_logging,
    logger,
)

RESULTS_URL_PATTERNS = [
    "https://www.flashscore.com/tennis/{tour_prefix}/{slug}-{year}/results/",
    "https://www.flashscore.com/tennis/{tour_prefix}/{slug}/results/?seasonId={year}",
    "https://www.flashscore.com/tennis/{tour_prefix}/{slug}/results/",
]


def _sleep(min_seconds: float = 0.6, max_seconds: float = 1.2) -> None:
    time.sleep(random.uniform(min_seconds, max_seconds))


def _stop_page_load(driver: Any) -> None:
    try:
        driver.execute_script("window.stop()")
    except Exception:
        pass


def _safe_get(driver: Any, url: str) -> None:
    try:
        driver.get(url)
    except TimeoutException:
        logger.warning("Timed out loading %s; stopping page load and continuing", url)
        _stop_page_load(driver)


def _accept_cookies(driver: Any) -> None:
    try:
        buttons = driver.find_elements(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
        if buttons:
            buttons[0].click()
            _sleep(0.5, 1.0)
    except Exception:
        pass


def _click_show_more(driver: Any) -> int:
    clicks = 0
    while True:
        buttons = driver.find_elements(By.CSS_SELECTOR, "a.event__more, div.event__more")
        if not buttons:
            break
        clicked = False
        for button in buttons:
            try:
                if button.is_displayed() and button.is_enabled():
                    driver.execute_script("arguments[0].click();", button)
                    _sleep(0.6, 1.0)
                    clicks += 1
                    clicked = True
                    break
            except Exception:
                continue
        if not clicked:
            break
    return clicks


def _extract_match_id(source_url: str, row_id: str) -> str:
    parsed = urlparse(source_url)
    mid = parse_qs(parsed.query).get("mid", [""])[0].strip()
    if mid:
        return mid
    if "_" in row_id:
        return row_id.split("_")[-1]
    return row_id.strip()


def _parse_event_datetime(raw_text: str, fallback_year: int) -> tuple[str | None, str | None]:
    text = str(raw_text or "").strip()
    if not text:
        return None, None

    parts = [part.strip() for part in text.split()]
    if not parts:
        return None, None

    date_part = parts[0]
    try:
        parsed = datetime.strptime(f"{date_part}{fallback_year}", "%d.%m.%Y")
        match_date = parsed.strftime("%Y-%m-%d")
    except Exception:
        match_date = None

    match_time = None
    if len(parts) > 1 and ":" in parts[1]:
        match_time = parts[1]

    return match_date, match_time


def _load_existing_match_ids() -> set[str]:
    if not ODDS_HISTORY_FILE.exists() or ODDS_HISTORY_FILE.stat().st_size == 0:
        return set()
    history = pd.read_csv(ODDS_HISTORY_FILE, usecols=["match_id"], low_memory=False)
    return {
        str(value).strip()
        for value in history["match_id"].astype("string").fillna("")
        if str(value).strip()
    }


def _load_feature_keys(tour: str, year: int) -> set[tuple[str, str, str]]:
    path = PROCESSED_DIR / f"{tour}_player_features.csv"
    if not path.exists():
        return set()

    feature_df = pd.read_csv(path, usecols=["match_date", "p1_id", "p2_id"], low_memory=False)
    feature_df["match_date"] = pd.to_datetime(feature_df["match_date"], errors="coerce")
    feature_df = feature_df[feature_df["match_date"].dt.year == year].copy()

    keys: set[tuple[str, str, str]] = set()
    for row in feature_df.itertuples(index=False):
        match_date = row.match_date.strftime("%Y-%m-%d") if pd.notna(row.match_date) else ""
        p1_id = _norm_player_id(row.p1_id)
        p2_id = _norm_player_id(row.p2_id)
        if not match_date or not p1_id or not p2_id:
            continue
        keys.add((match_date, min(p1_id, p2_id), max(p1_id, p2_id)))
    return keys


def _build_feature_player_pool(
    tour: str,
    year: int,
) -> tuple[list[str], dict[str, str], dict[tuple[str, str], list[tuple[str, str]]]]:
    path = PROCESSED_DIR / f"{tour}_player_features.csv"
    if not path.exists():
        return [], {}, {}

    usecols = ["match_date", "p1_name", "p2_name", "p1_id", "p2_id"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df[df["match_date"].dt.year == year].copy()

    pool: list[str] = []
    name_to_id: dict[str, str] = {}
    last_initial_index: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for name_col, id_col in (("p1_name", "p1_id"), ("p2_name", "p2_id")):
        for name, player_id in df[[name_col, id_col]].drop_duplicates().itertuples(index=False):
            name_text = str(name).strip()
            player_id_text = _norm_player_id(player_id)
            if not name_text or name_text.lower() == "nan" or not player_id_text:
                continue
            pool.append(name_text)
            name_to_id[name_text] = player_id_text

            parts = name_text.split()
            if len(parts) >= 2:
                first = parts[0]
                last = " ".join(parts[1:])
                key = (_normalize_name(last), _normalize_name(first)[:1])
                last_initial_index.setdefault(key, []).append((name_text, player_id_text))

    return sorted(set(pool)), name_to_id, last_initial_index


def _norm_player_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def _open_results_page(driver: Any, tour: str, slug: str, year: int) -> str | None:
    tour_prefix = "atp-singles" if tour == "atp" else "wta-singles"
    for pattern in RESULTS_URL_PATTERNS:
        url = pattern.format(tour_prefix=tour_prefix, slug=slug, year=year)
        try:
            logger.info("Opening results page: %s", url)
            _safe_get(driver, url)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.event__match"))
                )
            except TimeoutException:
                continue
            heading_year = ""
            headings = driver.find_elements(By.CSS_SELECTOR, ".heading__info")
            if headings:
                heading_year = (headings[0].text or "").strip()
            if heading_year and str(year) not in heading_year:
                continue
            matches = driver.find_elements(By.CSS_SELECTOR, "div.event__match")
            if matches:
                return url
        except Exception:
            continue
    return None


def _scrape_tournament_matches(
    driver: Any,
    tournament: dict[str, Any],
    tour: str,
    year: int,
    player_pool: list[str],
    name_to_id: dict[str, str],
    last_initial_index: dict[tuple[str, str], list[tuple[str, str]]],
    overrides: dict[str, str],
    feature_keys: set[tuple[str, str, str]],
    existing_match_ids: set[str],
    only_matchable: bool,
    max_matches: int | None,
) -> list[dict[str, Any]]:
    opened_url = _open_results_page(driver, tour=tour, slug=tournament["slug"], year=year)
    if opened_url is None:
        logger.warning("Could not open Flashscore results for %s %s", tour.upper(), tournament["slug"])
        return []

    clicks = _click_show_more(driver)
    if clicks:
        logger.info("Expanded results %d time(s) for %s", clicks, tournament["slug"])

    rows: list[dict[str, Any]] = []
    tournament_dt = datetime.strptime(str(tournament["tourney_date"]), "%Y%m%d")
    tournament_match_date = tournament_dt.strftime("%Y-%m-%d")
    tournament_week_match_date = (tournament_dt - timedelta(days=tournament_dt.weekday())).strftime("%Y-%m-%d")
    elements = driver.find_elements(By.CSS_SELECTOR, "div.event__match")
    error_samples = 0
    for element in elements:
        try:
            home_name_raw = element.find_element(By.CSS_SELECTOR, "div.event__participant--home").text.strip()
            away_name_raw = element.find_element(By.CSS_SELECTOR, "div.event__participant--away").text.strip()
            if not home_name_raw or not away_name_raw or "/" in home_name_raw or "/" in away_name_raw:
                continue

            source_url = element.find_element(By.CSS_SELECTOR, "a.eventRowLink").get_attribute("href") or ""
            row_id = element.get_attribute("id") or ""
            match_id = _extract_match_id(source_url, row_id)
            if not match_id or match_id in existing_match_ids:
                continue

            match_date, match_time = _parse_event_datetime(
                element.find_element(By.CSS_SELECTOR, "div.event__time").text,
                fallback_year=year,
            )
            if not match_date:
                continue

            resolved_p1, p1_id, p1_score = _resolve_player_name(
                home_name_raw, player_pool, name_to_id, last_initial_index, overrides
            )
            resolved_p2, p2_id, p2_score = _resolve_player_name(
                away_name_raw, player_pool, name_to_id, last_initial_index, overrides
            )

            stored_match_date = match_date
            if only_matchable:
                player_min = min(_norm_player_id(p1_id), _norm_player_id(p2_id))
                player_max = max(_norm_player_id(p1_id), _norm_player_id(p2_id))
                feature_key_actual = (match_date, player_min, player_max)
                feature_key_tournament = (tournament_match_date, player_min, player_max)
                feature_key_tournament_week = (tournament_week_match_date, player_min, player_max)
                if not p1_id or not p2_id or (
                    feature_key_actual not in feature_keys
                    and feature_key_tournament not in feature_keys
                    and feature_key_tournament_week not in feature_keys
                ):
                    continue
                if feature_key_tournament in feature_keys:
                    stored_match_date = tournament_match_date
                elif feature_key_tournament_week in feature_keys:
                    stored_match_date = tournament_week_match_date

            rows.append(
                {
                    "match_date": stored_match_date,
                    "match_time": match_time,
                    "tour": tour,
                    "tournament": tournament["name"],
                    "surface": tournament["surface"],
                    "player_1": home_name_raw,
                    "player_2": away_name_raw,
                    "player_1_resolved": resolved_p1,
                    "player_2_resolved": resolved_p2,
                    "player_1_id": p1_id,
                    "player_2_id": p2_id,
                    "player_1_match_score": p1_score,
                    "player_2_match_score": p2_score,
                    "source_url": source_url,
                    "match_id": match_id,
                }
            )
            if max_matches is not None and len(rows) >= max_matches:
                break
        except Exception as exc:
            if error_samples < 3:
                logger.warning("Row parse failed for %s %s: %s", tour.upper(), tournament["slug"], exc)
                error_samples += 1
            continue

    return rows


def _scrape_match_odds(driver: Any, source_url: str) -> tuple[float | None, float | None, int]:
    return _scrape_match_market_odds(driver, source_url)


def _append_history(rows: list[dict[str, Any]]) -> tuple[int, int]:
    history_path = ODDS_HISTORY_FILE
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.exists() and history_path.stat().st_size > 0:
        history = pd.read_csv(history_path, low_memory=False)
    else:
        history = pd.DataFrame(columns=OUTPUT_COLUMNS)

    for col in OUTPUT_COLUMNS:
        if col not in history.columns:
            history[col] = pd.NA

    before = int(len(history))
    if not rows:
        return before, before

    new_df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    new_df = new_df[OUTPUT_COLUMNS].copy()

    combined = pd.concat([history[OUTPUT_COLUMNS], new_df], ignore_index=True)
    combined["match_id"] = combined["match_id"].astype("string").fillna("")
    with_match_id = combined[combined["match_id"] != ""].copy()
    without_match_id = combined[combined["match_id"] == ""].copy()
    with_match_id = with_match_id.drop_duplicates(subset=["match_id"], keep="last")
    combined = pd.concat([without_match_id, with_match_id], ignore_index=True)
    combined.to_csv(history_path, index=False)
    after = int(len(combined))
    return before, after


def backfill_historical_odds(
    tours: tuple[str, ...],
    years: tuple[int, ...],
    tournament_slugs: tuple[str, ...] = (),
    limit_tournaments: int | None = None,
    max_matches_per_tournament: int | None = None,
    only_matchable: bool = True,
    page_load_timeout: int = 30,
) -> dict[str, Any]:
    _setup_logging()
    player_pool, name_to_id, last_initial_index = _build_player_pool()
    overrides = _load_name_overrides()
    existing_match_ids = _load_existing_match_ids()

    driver = _init_driver()
    driver.set_page_load_timeout(page_load_timeout)
    driver.set_script_timeout(15)
    _accept_cookies(driver)
    captured_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_rows: list[dict[str, Any]] = []
    tournament_reports: list[dict[str, Any]] = []
    scraped_matches = 0
    skipped_no_odds = 0

    try:
        for tour in tours:
            for year in years:
                registry = list(ALL_REGISTRIES.get((tour, year), []))
                if tournament_slugs:
                    registry = [item for item in registry if item["slug"] in tournament_slugs]
                if limit_tournaments is not None:
                    registry = registry[:limit_tournaments]

                feature_keys = _load_feature_keys(tour=tour, year=year) if only_matchable else set()
                feature_player_pool, feature_name_to_id, feature_last_initial_index = _build_feature_player_pool(
                    tour=tour,
                    year=year,
                )
                resolver_pool = feature_player_pool or player_pool
                resolver_name_to_id = feature_name_to_id or name_to_id
                resolver_last_initial_index = feature_last_initial_index or last_initial_index
                logger.info(
                    "Backfilling %s %s across %d tournament(s) with %d feature keys and %d resolver names",
                    tour.upper(),
                    year,
                    len(registry),
                    len(feature_keys),
                    len(resolver_pool),
                )

                for tournament in registry:
                    logger.info("Tournament: %s %s", tour.upper(), tournament["name"])
                    matches: list[dict[str, Any]] = []
                    tournament_failed = False
                    for attempt in range(1, 3):
                        try:
                            matches = _scrape_tournament_matches(
                                driver=driver,
                                tournament=tournament,
                                tour=tour,
                                year=year,
                                player_pool=resolver_pool,
                                name_to_id=resolver_name_to_id,
                                last_initial_index=resolver_last_initial_index,
                                overrides=overrides,
                                feature_keys=feature_keys,
                                existing_match_ids=existing_match_ids,
                                only_matchable=only_matchable,
                                max_matches=max_matches_per_tournament,
                            )
                            break
                        except (TimeoutException, WebDriverException) as exc:
                            tournament_failed = True
                            logger.warning(
                                "Tournament scrape failed for %s %s on attempt %d/2: %s",
                                tour.upper(),
                                tournament["slug"],
                                attempt,
                                exc,
                            )
                            _stop_page_load(driver)
                    else:
                        logger.warning(
                            "Skipping %s %s after 2 failed attempts",
                            tour.upper(),
                            tournament["slug"],
                        )

                    if tournament_failed and not matches:
                        tournament_reports.append(
                            {
                                "tour": tour,
                                "year": year,
                                "slug": tournament["slug"],
                                "name": tournament["name"],
                                "matches_considered": 0,
                                "rows_added": 0,
                            }
                        )
                        continue

                    added_for_tournament = 0
                    for row in matches:
                        odds_p1, odds_p2, bookmaker_count = _scrape_match_odds(driver, row["source_url"])
                        if odds_p1 is None or odds_p2 is None:
                            skipped_no_odds += 1
                            continue

                        all_rows.append(
                            {
                                **row,
                                "odds_p1": odds_p1,
                                "odds_p2": odds_p2,
                                "bookmaker": "flashscore_market_median",
                                "bookmaker_count": bookmaker_count,
                                "aggregation_method": "median",
                                "captured_at": captured_at,
                            }
                        )
                        existing_match_ids.add(str(row["match_id"]))
                        scraped_matches += 1
                        added_for_tournament += 1
                        logger.info(
                            "  Added odds for %s vs %s on %s (%d bookmakers)",
                            row["player_1_resolved"] or row["player_1"],
                            row["player_2_resolved"] or row["player_2"],
                            row["match_date"],
                            bookmaker_count,
                        )

                    tournament_reports.append(
                        {
                            "tour": tour,
                            "year": year,
                            "slug": tournament["slug"],
                            "name": tournament["name"],
                            "matches_considered": len(matches),
                            "rows_added": added_for_tournament,
                        }
                    )
    finally:
        driver.quit()

    before, after = _append_history(all_rows)
    return {
        "success": True,
        "captured_at": captured_at,
        "rows_scraped": scraped_matches,
        "rows_added": max(after - before, 0),
        "history_rows_before": before,
        "history_rows_after": after,
        "rows_skipped_no_odds": skipped_no_odds,
        "tournaments": tournament_reports,
        "history_file": str(ODDS_HISTORY_FILE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical tennis odds from Flashscore archive pages.")
    parser.add_argument("--tours", nargs="+", choices=["atp", "wta"], default=["atp", "wta"], help="Tours to backfill")
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026], help="Seasons to backfill")
    parser.add_argument("--tournament-slugs", nargs="*", default=[], help="Optional subset of tournament slugs")
    parser.add_argument("--limit-tournaments", type=int, default=None, help="Optional max number of tournaments per tour/year")
    parser.add_argument("--max-matches-per-tournament", type=int, default=None, help="Optional cap on matched rows scraped per tournament")
    parser.add_argument(
        "--page-load-timeout",
        type=int,
        default=30,
        help="Page load timeout in seconds before stopping the current load and continuing",
    )
    parser.add_argument(
        "--include-non-matchable",
        action="store_true",
        help="Do not restrict scraping to matches that already exist in local feature data",
    )
    args = parser.parse_args()

    report = backfill_historical_odds(
        tours=tuple(args.tours),
        years=tuple(args.years),
        tournament_slugs=tuple(args.tournament_slugs),
        limit_tournaments=args.limit_tournaments,
        max_matches_per_tournament=args.max_matches_per_tournament,
        only_matchable=not args.include_non_matchable,
        page_load_timeout=args.page_load_timeout,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
