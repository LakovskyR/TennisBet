"""
TML-Database (Tennismylife) → Sackmann format converter.

Downloads ATP match data from https://github.com/Tennismylife/TML-Database
for 2025 and 2026, converts TML short IDs to Sackmann numeric IDs via
name matching, reorders columns to exact Sackmann schema, and writes
to data/raw/tennis_atp/atp_matches_{year}.csv.

NOTE: TML-Database is ATP-only.  WTA backfill uses src/backfill_matches.py.

Usage:
    python -m src.tml_ingest                  # default: 2025 + 2026
    python -m src.tml_ingest --years 2025
    python -m src.tml_ingest --years 2025 2026
    python -m src.tml_ingest --check-only     # dry run: show stats, no write
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fuzzywuzzy import process as fuzz_process

from config import RAW_ATP, META_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TML_BASE_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
TML_PLAYER_DB_URL = f"{TML_BASE_URL}/ATP_Database.csv"

SACKMANN_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry", "winner_name",
    "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_seed", "loser_entry", "loser_name",
    "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
]

# Columns TML has but Sackmann doesn't
TML_EXTRA_COLS = ["indoor"]

ID_CACHE_FILE = META_DIR / "tml_id_cache.json"

FUZZY_THRESHOLD = 88


# ---------------------------------------------------------------------------
# Player ID Mapping: TML short-ID → Sackmann numeric ID
# ---------------------------------------------------------------------------
def _build_sackmann_name_index() -> dict[str, str]:
    """Load Sackmann atp_players.csv → {normalized_name: player_id_str}."""
    path = RAW_ATP / "atp_players.csv"
    if not path.exists():
        log.warning(f"Sackmann player file not found: {path}")
        return {}
    df = pd.read_csv(path, usecols=["player_id", "name_first", "name_last"], low_memory=False)
    index: dict[str, str] = {}
    for pid, first, last in df.itertuples(index=False):
        full = f"{first} {last}".strip()
        if not full or full.lower() == "nan":
            continue
        try:
            sid = str(int(float(pid)))
        except (ValueError, TypeError):
            continue
        norm = _normalize(full)
        # Keep first occurrence (most common / senior player for duplicate names)
        if norm not in index:
            index[norm] = sid
    return index


def _normalize(name: str) -> str:
    """Lowercase, collapse whitespace, remove dots/hyphens."""
    return " ".join(str(name).lower().replace(".", " ").replace("-", " ").split())


def _load_id_cache() -> dict[str, str]:
    """Load cached TML-ID → Sackmann-ID mappings."""
    if ID_CACHE_FILE.exists():
        text = ID_CACHE_FILE.read_text(encoding="utf-8-sig")
        if text.strip():
            return json.loads(text)
    return {}


def _save_id_cache(cache: dict[str, str]) -> None:
    """Persist TML-ID → Sackmann-ID cache."""
    META_DIR.mkdir(parents=True, exist_ok=True)
    ID_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _build_tml_id_to_name(df: pd.DataFrame) -> dict[str, str]:
    """Extract unique TML ID → player name mapping from match dataframe."""
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        wid = str(row.get("winner_id", "")).strip()
        wname = str(row.get("winner_name", "")).strip()
        lid = str(row.get("loser_id", "")).strip()
        lname = str(row.get("loser_name", "")).strip()
        if wid and wname and wname.lower() != "nan":
            mapping[wid] = wname
        if lid and lname and lname.lower() != "nan":
            mapping[lid] = lname
    return mapping


def _resolve_tml_ids(
    tml_id_to_name: dict[str, str],
    sackmann_index: dict[str, str],
    cache: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Map TML short IDs to Sackmann numeric IDs.

    Returns:
        (tml_id → sackmann_id mapping, unresolved {tml_id: name})
    """
    resolved: dict[str, str] = {}
    unresolved: dict[str, str] = {}
    sackmann_names = list(sackmann_index.keys())

    for tml_id, name in tml_id_to_name.items():
        # Check cache first
        if tml_id in cache:
            resolved[tml_id] = cache[tml_id]
            continue

        norm = _normalize(name)

        # Exact match
        if norm in sackmann_index:
            resolved[tml_id] = sackmann_index[norm]
            cache[tml_id] = sackmann_index[norm]
            continue

        # Fuzzy match
        if sackmann_names:
            best = fuzz_process.extractOne(norm, sackmann_names)
            if best and best[1] >= FUZZY_THRESHOLD:
                sid = sackmann_index[best[0]]
                resolved[tml_id] = sid
                cache[tml_id] = sid
                if best[1] < 95:
                    log.debug(f"  Fuzzy: '{name}' → '{best[0]}' (score={best[1]}, id={sid})")
                continue

        unresolved[tml_id] = name

    return resolved, unresolved


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_tml_csv(year: int) -> pd.DataFrame:
    """Download TML match CSV for a given year from GitHub."""
    url = f"{TML_BASE_URL}/{year}.csv"
    log.info(f"Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    log.info(f"  Downloaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Convert TML → Sackmann Format
# ---------------------------------------------------------------------------
def convert_to_sackmann(
    df: pd.DataFrame,
    id_map: dict[str, str],
) -> pd.DataFrame:
    """
    Convert TML DataFrame to Sackmann schema.

    Steps:
    1. Replace TML IDs with Sackmann IDs
    2. Drop extra columns (indoor)
    3. Reorder columns to Sackmann order
    """
    out = df.copy()

    # Replace IDs
    out["winner_id"] = out["winner_id"].astype(str).map(id_map)
    out["loser_id"] = out["loser_id"].astype(str).map(id_map)

    # Drop rows where we couldn't resolve either player
    before = len(out)
    out = out.dropna(subset=["winner_id", "loser_id"])
    dropped = before - len(out)
    if dropped > 0:
        log.info(f"  Dropped {dropped} rows with unresolved player IDs")

    # Convert IDs to int strings (no .0)
    out["winner_id"] = out["winner_id"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "")
    out["loser_id"] = out["loser_id"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "")

    # Drop TML-only columns
    for col in TML_EXTRA_COLS:
        if col in out.columns:
            out = out.drop(columns=[col])

    # Ensure all Sackmann columns exist
    for col in SACKMANN_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    # Reorder to exact Sackmann column order
    out = out[SACKMANN_COLUMNS]

    # Sort by date, tourney, match
    out = out.sort_values(
        by=["tourney_date", "tourney_id", "match_num"],
        ascending=True,
    ).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Write Output
# ---------------------------------------------------------------------------
def write_output(
    df: pd.DataFrame,
    year: int,
    incremental: bool = True,
) -> Path:
    """Write converted DataFrame to Sackmann-format CSV."""
    RAW_ATP.mkdir(parents=True, exist_ok=True)
    output_path = RAW_ATP / f"atp_matches_{year}.csv"

    if incremental and output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path, low_memory=False)
        log.info(f"  Existing file: {len(existing)} rows")

        # Merge: keep TML data as authoritative (has richer stats)
        combined = pd.concat([existing, df], ignore_index=True)

        # Dedup: tourney_id + match_num + winner_id + loser_id
        combined["_dedup"] = (
            combined["tourney_id"].astype(str) + "|"
            + combined["match_num"].astype(str) + "|"
            + combined["winner_id"].astype(str) + "|"
            + combined["loser_id"].astype(str)
        )
        combined = combined.drop_duplicates(subset=["_dedup"], keep="last")
        combined = combined.drop(columns=["_dedup"])
        df = combined

    df = df.sort_values(
        by=["tourney_date", "tourney_id", "match_num"],
        ascending=True,
    ).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    log.info(f"  Written {len(df)} rows → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(path: Path) -> dict[str, Any]:
    """Quick validation of output file."""
    if not path.exists():
        return {"valid": False, "error": "File missing"}
    df = pd.read_csv(path, low_memory=False)
    issues = []

    # Column check
    missing_cols = set(SACKMANN_COLUMNS) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Duplicate check
    df["_key"] = (
        df["tourney_id"].astype(str) + "|"
        + df["match_num"].astype(str) + "|"
        + df["winner_id"].astype(str) + "|"
        + df["loser_id"].astype(str)
    )
    dupes = df.duplicated(subset=["_key"]).sum()
    if dupes > 0:
        issues.append(f"{dupes} duplicate rows")

    # ID format check (should be numeric)
    non_numeric_w = df["winner_id"].astype(str).str.contains(r"[^0-9.]", na=False).sum()
    non_numeric_l = df["loser_id"].astype(str).str.contains(r"[^0-9.]", na=False).sum()
    if non_numeric_w > 0 or non_numeric_l > 0:
        issues.append(f"Non-numeric IDs: {non_numeric_w} winner, {non_numeric_l} loser")

    tournaments = df["tourney_id"].nunique()
    date_range = f"{df['tourney_date'].min()} - {df['tourney_date'].max()}"

    return {
        "valid": len(issues) == 0,
        "rows": len(df),
        "tournaments": tournaments,
        "date_range": date_range,
        "issues": issues,
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def ingest(
    years: list[int] | None = None,
    check_only: bool = False,
    incremental: bool = True,
) -> dict[str, Any]:
    """
    Download TML data, convert to Sackmann format, write to raw dir.

    Args:
        years: Which years to download (default: [2025, 2026])
        check_only: If True, download and report but don't write files
        incremental: If True, merge with existing CSV

    Returns:
        Report dict
    """
    if years is None:
        years = [2025, 2026]

    log.info("=" * 60)
    log.info("TML-Database → Sackmann Ingest")
    log.info(f"Years: {years}")
    log.info("=" * 60)

    # Build Sackmann player name index
    log.info("Building Sackmann player name index...")
    sackmann_index = _build_sackmann_name_index()
    log.info(f"  {len(sackmann_index)} unique player names")

    # Load ID cache
    cache = _load_id_cache()
    log.info(f"  ID cache: {len(cache)} entries")

    reports: dict[str, Any] = {}

    for year in years:
        log.info(f"\n{'─' * 40}")
        log.info(f"Processing {year}")
        log.info(f"{'─' * 40}")

        try:
            # Download
            tml_df = download_tml_csv(year)
            if tml_df.empty:
                reports[str(year)] = {"success": False, "error": "Empty CSV"}
                continue

            # Build TML ID → name mapping from this year's data
            tml_id_to_name = _build_tml_id_to_name(tml_df)
            log.info(f"  {len(tml_id_to_name)} unique TML player IDs")

            # Resolve TML IDs to Sackmann IDs
            id_map, unresolved = _resolve_tml_ids(tml_id_to_name, sackmann_index, cache)
            log.info(f"  Resolved: {len(id_map)} players")
            if unresolved:
                log.warning(f"  Unresolved: {len(unresolved)} players:")
                for uid, uname in sorted(unresolved.items()):
                    log.warning(f"    {uid}: {uname}")

            # Convert
            sackmann_df = convert_to_sackmann(tml_df, id_map)
            log.info(f"  Converted: {len(sackmann_df)} rows in Sackmann format")

            if check_only:
                log.info("  [CHECK-ONLY] Skipping file write")
                reports[str(year)] = {
                    "success": True,
                    "rows_downloaded": len(tml_df),
                    "rows_converted": len(sackmann_df),
                    "players_resolved": len(id_map),
                    "players_unresolved": len(unresolved),
                    "unresolved_names": list(unresolved.values()),
                    "check_only": True,
                }
                continue

            # Write
            output_path = write_output(sackmann_df, year, incremental=incremental)

            # Validate
            val = validate(output_path)

            reports[str(year)] = {
                "success": True,
                "rows_downloaded": len(tml_df),
                "rows_converted": len(sackmann_df),
                "rows_written": val["rows"],
                "tournaments": val["tournaments"],
                "date_range": val["date_range"],
                "players_resolved": len(id_map),
                "players_unresolved": len(unresolved),
                "unresolved_names": list(unresolved.values())[:20],
                "validation": val,
                "output_file": str(output_path),
            }

        except requests.HTTPError as e:
            log.error(f"  HTTP error: {e}")
            reports[str(year)] = {"success": False, "error": str(e)}
        except Exception as e:
            log.error(f"  Error: {e}")
            reports[str(year)] = {"success": False, "error": str(e)}

    # Save updated ID cache
    _save_id_cache(cache)
    log.info(f"\nID cache updated: {len(cache)} entries → {ID_CACHE_FILE}")

    log.info(f"\n{'=' * 60}")
    log.info("INGEST COMPLETE")
    for yr, rep in reports.items():
        status = "OK" if rep.get("success") else "FAILED"
        rows = rep.get("rows_written") or rep.get("rows_converted", 0)
        log.info(f"  {yr}: {status} ({rows} rows)")
    log.info(f"{'=' * 60}")

    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TML-Database → Sackmann format converter (ATP)")
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026],
                        help="Years to download (default: 2025 2026)")
    parser.add_argument("--check-only", action="store_true",
                        help="Download and report but don't write files")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Overwrite existing files instead of merging")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    report = ingest(
        years=args.years,
        check_only=args.check_only,
        incremental=not args.no_incremental,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
