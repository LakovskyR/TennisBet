from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import ODDS_MOVEMENT_FILE
from src.sqlite_storage import load_odds_frame

MOVEMENT_COLUMNS = [
    "match_date",
    "tour",
    "player_1",
    "player_2",
    "match_id",
    "captured_at",
    "previous_captured_at",
    "hours_since_previous_snapshot",
    "odds_p1",
    "odds_p2",
    "prev_odds_p1",
    "prev_odds_p2",
    "odds_p1_delta",
    "odds_p2_delta",
    "odds_p1_pct_change",
    "odds_p2_pct_change",
    "market_move",
]


def _movement_key(row: pd.Series) -> str:
    match_id = str(row.get("match_id") or "").strip()
    if match_id:
        return match_id
    return "|".join(
        [
            str(row.get("match_date") or ""),
            str(row.get("tour") or "").lower(),
            str(row.get("player_1_resolved") or row.get("player_1") or "").lower(),
            str(row.get("player_2_resolved") or row.get("player_2") or "").lower(),
        ]
    )


def _market_move(delta_p1: float | None, delta_p2: float | None) -> str:
    if delta_p1 is None or delta_p2 is None:
        return "no_prior_snapshot"
    eps = 1e-9
    if abs(delta_p1) < eps and abs(delta_p2) < eps:
        return "flat"
    if delta_p1 < 0 and delta_p2 > 0:
        return "steam_p1"
    if delta_p2 < 0 and delta_p1 > 0:
        return "steam_p2"
    return "mixed"


def build_odds_movement_frame(
    *,
    current_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    current = current_df.copy() if current_df is not None else load_odds_frame(current_only=True, fallback_to_csv=False)
    history = history_df.copy() if history_df is not None else load_odds_frame(current_only=False, fallback_to_csv=False)
    if current.empty:
        return pd.DataFrame(columns=MOVEMENT_COLUMNS)

    for df in (current, history):
        if "captured_at" in df.columns:
            df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce", utc=True)
        else:
            df["captured_at"] = pd.NaT
        df["_movement_key"] = df.apply(_movement_key, axis=1)
        df["odds_p1"] = pd.to_numeric(df.get("odds_p1"), errors="coerce")
        df["odds_p2"] = pd.to_numeric(df.get("odds_p2"), errors="coerce")

    history = history.sort_values(["_movement_key", "captured_at"])
    rows: list[dict[str, Any]] = []

    for _, row in current.sort_values(["match_date", "tour", "player_1", "player_2"]).iterrows():
        key = row["_movement_key"]
        current_ts = row.get("captured_at")
        prior = history[
            (history["_movement_key"] == key)
            & history["captured_at"].notna()
            & (history["captured_at"] < current_ts)
        ]
        prev_row = prior.iloc[-1] if not prior.empty else None

        prev_odds_p1 = float(prev_row["odds_p1"]) if prev_row is not None and pd.notna(prev_row["odds_p1"]) else None
        prev_odds_p2 = float(prev_row["odds_p2"]) if prev_row is not None and pd.notna(prev_row["odds_p2"]) else None
        curr_odds_p1 = float(row["odds_p1"]) if pd.notna(row["odds_p1"]) else None
        curr_odds_p2 = float(row["odds_p2"]) if pd.notna(row["odds_p2"]) else None

        delta_p1 = (curr_odds_p1 - prev_odds_p1) if curr_odds_p1 is not None and prev_odds_p1 is not None else None
        delta_p2 = (curr_odds_p2 - prev_odds_p2) if curr_odds_p2 is not None and prev_odds_p2 is not None else None
        pct_p1 = (delta_p1 / prev_odds_p1) if delta_p1 is not None and prev_odds_p1 not in (None, 0) else None
        pct_p2 = (delta_p2 / prev_odds_p2) if delta_p2 is not None and prev_odds_p2 not in (None, 0) else None
        previous_captured_at = prev_row["captured_at"] if prev_row is not None else pd.NaT
        hours_since_previous = None
        if pd.notna(current_ts) and pd.notna(previous_captured_at):
            hours_since_previous = float((current_ts - previous_captured_at).total_seconds() / 3600.0)

        rows.append(
            {
                "match_date": row.get("match_date"),
                "tour": row.get("tour"),
                "player_1": row.get("player_1"),
                "player_2": row.get("player_2"),
                "match_id": row.get("match_id"),
                "captured_at": current_ts.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(current_ts) else None,
                "previous_captured_at": previous_captured_at.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(previous_captured_at) else None,
                "hours_since_previous_snapshot": hours_since_previous,
                "odds_p1": curr_odds_p1,
                "odds_p2": curr_odds_p2,
                "prev_odds_p1": prev_odds_p1,
                "prev_odds_p2": prev_odds_p2,
                "odds_p1_delta": delta_p1,
                "odds_p2_delta": delta_p2,
                "odds_p1_pct_change": pct_p1,
                "odds_p2_pct_change": pct_p2,
                "market_move": _market_move(delta_p1, delta_p2),
            }
        )

    movement = pd.DataFrame(rows)
    if movement.empty:
        return pd.DataFrame(columns=MOVEMENT_COLUMNS)
    for col in MOVEMENT_COLUMNS:
        if col not in movement.columns:
            movement[col] = pd.NA
    return movement[MOVEMENT_COLUMNS]


def write_odds_movement_snapshot(
    *,
    output_path: Path = ODDS_MOVEMENT_FILE,
    current_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    movement = build_odds_movement_frame(current_df=current_df, history_df=history_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movement.to_csv(output_path, index=False)
    return {
        "output_path": str(output_path),
        "rows": int(len(movement)),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
