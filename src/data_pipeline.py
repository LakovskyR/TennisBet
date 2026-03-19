from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    CUSTOM_ATP_FILE,
    CUSTOM_WTA_FILE,
    DATE_FMT,
    LAST_UPDATE_FILE,
    PROCESSED_DIR,
    RAW_ATP,
    RAW_WTA,
)
from src.data_updater import get_staleness_status
from src.sqlite_storage import initialize_database, sync_matches_frame, sync_player_aliases, sync_reference_players

TOUR_RAW = {
    "atp": RAW_ATP,
    "wta": RAW_WTA,
}

TOUR_CUSTOM = {
    "atp": CUSTOM_ATP_FILE,
    "wta": CUSTOM_WTA_FILE,
}

LEVEL_MAP = {
    "G": "grand_slam",
    "M": "masters",
    "A": "tour",
    "C": "challenger",
    "D": "davis_or_equivalent",
    "F": "finals",
    "I": "team_event",
}

SCORE_RETIREMENT_MARKERS = ("RET", "W/O", "DEF", "ABN", "unfinished", "Walkover")
MATCH_SYNC_TEXT_COLUMNS = [
    "match_date",
    "tourney_id",
    "tourney_name",
    "surface",
    "tournament_level",
    "source",
    "source_file",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "loser_id",
    "loser_name",
    "score",
    "best_of",
    "round",
]
MATCH_SYNC_INT_COLUMNS = [
    "draw_size",
    "winner_sets_won",
    "loser_sets_won",
    "total_games",
    "year",
    "days_since_epoch",
]
MATCH_SYNC_BOOL_COLUMNS = [
    "is_retirement",
    "is_walkover",
    "is_training_eligible",
]


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LAST_UPDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    initialize_database()


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame()

    base = usable[0]
    aligned = [base]
    for frame in usable[1:]:
        trimmed = frame.loc[:, ~frame.isna().all()].copy()
        aligned.append(trimmed.reindex(columns=base.columns))
    return pd.concat(aligned, ignore_index=True)


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _normalize_int_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").round().astype("Int64")
    return numeric.astype("string").fillna("")


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip().str.lower()
    mapped = text.map(
        {
            "": "",
            "1": "1",
            "0": "0",
            "true": "1",
            "false": "0",
            "yes": "1",
            "no": "0",
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_text = numeric.round().astype("Int64").astype("string").fillna("")
    return mapped.fillna(numeric_text).replace({"<NA>": ""})


def _rows_requiring_match_sync(existing: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return combined.copy()
    if existing.empty or "match_key" not in existing.columns:
        return combined.drop_duplicates(subset=["match_key"], keep="last").copy()

    current_latest = combined.drop_duplicates(subset=["match_key"], keep="last").copy()
    existing_latest = existing.drop_duplicates(subset=["match_key"], keep="last").copy()

    compare_columns = MATCH_SYNC_TEXT_COLUMNS + MATCH_SYNC_INT_COLUMNS + MATCH_SYNC_BOOL_COLUMNS
    for col in compare_columns:
        if col not in current_latest.columns:
            current_latest[col] = pd.NA
        if col not in existing_latest.columns:
            existing_latest[col] = pd.NA

    existing_compare = existing_latest[["match_key", *compare_columns]].copy()
    current_compare = current_latest[["match_key", *compare_columns]].copy()
    merged = current_compare.merge(
        existing_compare,
        on="match_key",
        how="left",
        suffixes=("_new", "_old"),
        indicator=True,
    )

    changed_mask = merged["_merge"].eq("left_only")
    for col in MATCH_SYNC_TEXT_COLUMNS:
        changed_mask |= _normalize_text_series(merged[f"{col}_new"]) != _normalize_text_series(merged[f"{col}_old"])
    for col in MATCH_SYNC_INT_COLUMNS:
        changed_mask |= _normalize_int_series(merged[f"{col}_new"]) != _normalize_int_series(merged[f"{col}_old"])
    for col in MATCH_SYNC_BOOL_COLUMNS:
        changed_mask |= _normalize_bool_series(merged[f"{col}_new"]) != _normalize_bool_series(merged[f"{col}_old"])

    sync_keys = set(merged.loc[changed_mask, "match_key"].astype("string").tolist())
    return current_latest[current_latest["match_key"].astype("string").isin(sync_keys)].copy()


def _yearly_match_files(raw_repo: Path, tour: str) -> list[Path]:
    prefix = f"{tour}_matches_"
    files = [
        p
        for p in raw_repo.glob(f"{prefix}*.csv")
        if p.name.replace(prefix, "").replace(".csv", "").isdigit()
    ]
    return sorted(files)


def _parse_score(score: str) -> dict[str, Any]:
    if not isinstance(score, str) or not score.strip():
        return {
            "winner_sets_won": None,
            "loser_sets_won": None,
            "total_games": None,
            "is_retirement": False,
        }

    score_clean = score.strip()
    is_ret = any(marker.lower() in score_clean.lower() for marker in SCORE_RETIREMENT_MARKERS)

    winner_sets = 0
    loser_sets = 0
    total_games = 0

    tokens = score_clean.split()
    for token in tokens:
        if "-" not in token:
            continue
        base = token.split("(")[0]
        parts = base.split("-")
        if len(parts) != 2:
            continue
        if not parts[0].isdigit() or not parts[1].isdigit():
            continue

        w_games = int(parts[0])
        l_games = int(parts[1])
        total_games += w_games + l_games
        if w_games > l_games:
            winner_sets += 1
        elif l_games > w_games:
            loser_sets += 1

    return {
        "winner_sets_won": winner_sets if winner_sets or loser_sets else None,
        "loser_sets_won": loser_sets if winner_sets or loser_sets else None,
        "total_games": total_games if total_games else None,
        "is_retirement": is_ret,
    }


def _load_and_concat_raw_matches(tour: str) -> pd.DataFrame:
    files = _yearly_match_files(TOUR_RAW[tour], tour)
    if not files:
        raise FileNotFoundError(f"No yearly files found for {tour}: {TOUR_RAW[tour]}")

    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        if frame.empty:
            continue
        frame["source"] = "sackmann"
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No readable raw match files for {tour}.")

    return pd.concat(frames, ignore_index=True)


def _load_custom_matches(tour: str, expected_columns: list[str]) -> pd.DataFrame:
    custom_file = TOUR_CUSTOM[tour]
    if not custom_file.exists() or custom_file.stat().st_size == 0:
        return pd.DataFrame(columns=expected_columns)

    custom_df = pd.read_csv(custom_file, low_memory=False)
    if custom_df.empty:
        return pd.DataFrame(columns=expected_columns)

    custom_df["source"] = "custom"
    for col in expected_columns:
        if col not in custom_df.columns:
            custom_df[col] = pd.NA
    return custom_df[expected_columns]


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["winner_id", "loser_id", "score"]
    df = df.dropna(subset=required).copy()

    if "winner_name" in df.columns:
        df = df[df["winner_name"].notna()]
    if "loser_name" in df.columns:
        df = df[df["loser_name"].notna()]

    if "tourney_date" not in df.columns:
        raise KeyError("Missing required column: tourney_date")

    df["match_date"] = pd.to_datetime(df["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce")
    df = df[df["match_date"].notna()].copy()

    parsed_scores = df["score"].astype("string").map(_parse_score).apply(pd.Series)
    for col in parsed_scores.columns:
        df[col] = parsed_scores[col]

    df["year"] = df["match_date"].dt.year.astype("Int64")
    df["days_since_epoch"] = (
        (df["match_date"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
    ).astype("int64")

    level_raw = df.get("tourney_level", pd.Series([pd.NA] * len(df), index=df.index)).astype("string")
    df["tournament_level"] = level_raw.map(LEVEL_MAP).fillna("other")

    df["is_walkover"] = df["score"].astype("string").str.contains("W/O", case=False, na=False)

    # Keep records but mark training eligibility.
    df["is_training_eligible"] = ~(df["is_retirement"].fillna(False) | df["is_walkover"].fillna(False))

    # Use a scrape-stable identity so reruns with different match_num/round formatting
    # still collapse to one match record.
    key_cols = ["tourney_id", "tourney_date", "winner_id", "loser_id", "score"]
    for col in key_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df["match_key"] = (
        df[key_cols]
        .astype("string")
        .fillna("NA")
        .agg("|".join, axis=1)
    )

    return df


def build_master_for_tour(tour: str, incremental: bool = True) -> dict[str, Any]:
    _ensure_dirs()

    raw_df = _load_and_concat_raw_matches(tour)
    expected_columns = list(raw_df.columns)

    custom_df = _load_custom_matches(tour, expected_columns)
    combined = _concat_frames([raw_df, custom_df]) if not custom_df.empty else raw_df.copy()
    combined = _derive_columns(combined)

    output_file = PROCESSED_DIR / f"{tour}_matches_master.csv"
    sync_df = combined.copy()

    if incremental and output_file.exists():
        existing = pd.read_csv(output_file, low_memory=False)
        if "match_key" not in existing.columns:
            existing = _derive_columns(existing)
        existing_keys = set(existing["match_key"].astype("string").fillna("").tolist())
        before = len(existing)
        merged = _concat_frames([existing, combined]) if not combined.empty else existing.copy()
        merged = merged.drop_duplicates(subset=["match_key"], keep="last")
        merged["match_date"] = pd.to_datetime(merged["match_date"], errors="coerce")
        merged = merged.sort_values(["match_date", "tourney_id", "match_num"], na_position="last")
        rows_added = max(0, len(merged) - before)
        final_df = merged
        sync_df = _rows_requiring_match_sync(existing, combined)
    else:
        final_df = combined.drop_duplicates(subset=["match_key"], keep="last")
        final_df = final_df.sort_values(["match_date", "tourney_id", "match_num"], na_position="last")
        rows_added = len(final_df)
        sync_df = final_df.copy()

    final_df["match_date"] = pd.to_datetime(final_df["match_date"], errors="coerce").dt.strftime(DATE_FMT)
    final_df.to_csv(output_file, index=False)
    if not sync_df.empty:
        sync_df["match_date"] = pd.to_datetime(sync_df["match_date"], errors="coerce").dt.strftime(DATE_FMT)
        sync_matches_frame(sync_df, tour=tour)

    max_date = pd.to_datetime(final_df["match_date"], errors="coerce").max()
    result = {
        "tour": tour,
        "output": str(output_file),
        "rows": int(len(final_df)),
        "rows_added": int(rows_added),
        "latest_match_date": max_date.strftime(DATE_FMT) if pd.notna(max_date) else None,
    }
    return result


def run_pipeline(incremental: bool = True) -> dict[str, Any]:
    _ensure_dirs()
    sync_reference_players()
    sync_player_aliases()

    results = {tour: build_master_for_tour(tour, incremental=incremental) for tour in ("atp", "wta")}

    if LAST_UPDATE_FILE.exists():
        text = LAST_UPDATE_FILE.read_text(encoding="utf-8-sig")
        state = json.loads(text) if text.strip() else {}
    else:
        state = {}

    latest_dates = [
        datetime.strptime(r["latest_match_date"], DATE_FMT).date()
        for r in results.values()
        if r.get("latest_match_date")
    ]
    latest_match = max(latest_dates) if latest_dates else None

    state.update(
        {
            "pipeline_last_run": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pipeline": results,
            "last_new_match": latest_match.strftime(DATE_FMT) if latest_match else None,
            "staleness": get_staleness_status(latest_match),
        }
    )
    LAST_UPDATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    return results


def main() -> None:
    results = run_pipeline(incremental=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
