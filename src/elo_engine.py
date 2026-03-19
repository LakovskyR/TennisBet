from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    DATE_FMT,
    ELO_K_BASE,
    ELO_START,
    LAST_UPDATE_FILE,
    PROCESSED_DIR,
    SURFACE_LIST,
    TOURNAMENT_WEIGHTS,
)
from src.sqlite_storage import load_matches_frame, sync_elo_ratings

ROUND_MULTIPLIER = {
    "R128": 0.95,
    "R64": 0.95,
    "R32": 1.00,
    "R16": 1.02,
    "QF": 1.05,
    "SF": 1.08,
    "F": 1.10,
}


def _expected_score(player_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))


def _k_factor(tourney_level: str | None, round_name: str | None, best_of: Any) -> float:
    level_mult = TOURNAMENT_WEIGHTS.get(str(tourney_level), 1.0)
    round_mult = ROUND_MULTIPLIER.get(str(round_name), 1.0)

    best_of_mult = 1.0
    try:
        if pd.notna(best_of) and float(best_of) >= 5:
            best_of_mult = 1.05
    except Exception:
        best_of_mult = 1.0

    return ELO_K_BASE * level_mult * round_mult * best_of_mult


def _load_matches(tour: str) -> pd.DataFrame:
    df = load_matches_frame(
        tour,
        columns=[
            "match_key",
            "tourney_id",
            "tourney_name",
            "surface",
            "tourney_level",
            "round",
            "best_of",
            "winner_id",
            "winner_name",
            "loser_id",
            "loser_name",
            "match_date",
            "match_num",
            "is_training_eligible",
        ],
        fallback_to_csv=False,
    )
    if df.empty:
        return df

    if "match_date" not in df.columns:
        raise KeyError(f"Missing match_date for {tour} match history")

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df[df["match_date"].notna()].copy()

    if "is_training_eligible" in df.columns:
        eligible = df["is_training_eligible"].astype("string").str.lower().isin(["true", "1"])
        # If parsed boolean values are absent, keep all rows.
        if eligible.any():
            df = df[eligible].copy()

    required_cols = [
        "match_key",
        "tourney_id",
        "tourney_name",
        "surface",
        "tourney_level",
        "round",
        "best_of",
        "winner_id",
        "winner_name",
        "loser_id",
        "loser_name",
        "match_date",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.sort_values(["match_date", "tourney_id", "match_num"], na_position="last")
    return df


def _init_ratings_from_existing(
    path: Path,
) -> tuple[dict[str, float], dict[tuple[str, str], float], set[str], pd.DataFrame]:
    if not path.exists():
        return {}, {}, set(), pd.DataFrame()

    existing = pd.read_csv(path, low_memory=False)
    if existing.empty:
        return {}, {}, set(), existing

    if "match_key" in existing.columns:
        processed_keys = set(existing["match_key"].dropna().astype(str).tolist())
    else:
        processed_keys = set()

    overall: dict[str, float] = {}
    surface: dict[tuple[str, str], float] = {}

    if "match_date" in existing.columns:
        existing["match_date"] = pd.to_datetime(existing["match_date"], errors="coerce")

    if {"player_id", "elo_post"}.issubset(existing.columns):
        latest_per_player = existing.dropna(subset=["player_id", "elo_post"]).sort_values("match_date")
        for _, row in latest_per_player.groupby("player_id", as_index=False).tail(1).iterrows():
            overall[str(row["player_id"])] = float(row["elo_post"])

    if {"player_id", "surface", "surface_elo_post"}.issubset(existing.columns):
        latest_surface = existing.dropna(subset=["player_id", "surface", "surface_elo_post"]).sort_values("match_date")
        for _, row in latest_surface.groupby(["player_id", "surface"], as_index=False).tail(1).iterrows():
            surface[(str(row["player_id"]), str(row["surface"]))] = float(row["surface_elo_post"])

    return overall, surface, processed_keys, existing


def compute_elo_for_tour(tour: str, incremental: bool = True) -> dict[str, Any]:
    matches = _load_matches(tour)
    output_path = PROCESSED_DIR / f"{tour}_elo_ratings.csv"

    overall_ratings: dict[str, float]
    surface_ratings: dict[tuple[str, str], float]
    processed_keys: set[str]
    existing_df: pd.DataFrame

    if incremental:
        overall_ratings, surface_ratings, processed_keys, existing_df = _init_ratings_from_existing(output_path)
    else:
        overall_ratings, surface_ratings, processed_keys, existing_df = {}, {}, set(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    processed_matches = 0

    for _, m in matches.iterrows():
        match_key = str(m.get("match_key", ""))
        if incremental and match_key and match_key in processed_keys:
            continue

        winner = str(m["winner_id"])
        loser = str(m["loser_id"])
        surface = str(m.get("surface") or "Unknown")

        w_overall_pre = overall_ratings.get(winner, ELO_START)
        l_overall_pre = overall_ratings.get(loser, ELO_START)

        w_surface_pre = surface_ratings.get((winner, surface), w_overall_pre)
        l_surface_pre = surface_ratings.get((loser, surface), l_overall_pre)

        combined_w_pre = 0.7 * w_overall_pre + 0.3 * w_surface_pre
        combined_l_pre = 0.7 * l_overall_pre + 0.3 * l_surface_pre

        expected_w = _expected_score(combined_w_pre, combined_l_pre)
        expected_l = 1.0 - expected_w

        k = _k_factor(m.get("tourney_level"), m.get("round"), m.get("best_of"))

        w_overall_post = w_overall_pre + k * (1.0 - expected_w)
        l_overall_post = l_overall_pre + k * (0.0 - expected_l)

        w_surface_post = w_surface_pre + k * (1.0 - expected_w)
        l_surface_post = l_surface_pre + k * (0.0 - expected_l)

        overall_ratings[winner] = w_overall_post
        overall_ratings[loser] = l_overall_post
        surface_ratings[(winner, surface)] = w_surface_post
        surface_ratings[(loser, surface)] = l_surface_post

        common = {
            "match_key": match_key,
            "match_date": m["match_date"].strftime(DATE_FMT),
            "tour": tour,
            "tourney_id": m.get("tourney_id"),
            "tourney_name": m.get("tourney_name"),
            "surface": surface,
            "tourney_level": m.get("tourney_level"),
            "round": m.get("round"),
            "k_factor": round(k, 6),
        }

        rows.append(
            {
                **common,
                "player_id": winner,
                "player_name": m.get("winner_name"),
                "opponent_id": loser,
                "opponent_name": m.get("loser_name"),
                "result": 1,
                "elo_pre": round(w_overall_pre, 6),
                "elo_post": round(w_overall_post, 6),
                "surface_elo_pre": round(w_surface_pre, 6),
                "surface_elo_post": round(w_surface_post, 6),
            }
        )
        rows.append(
            {
                **common,
                "player_id": loser,
                "player_name": m.get("loser_name"),
                "opponent_id": winner,
                "opponent_name": m.get("winner_name"),
                "result": 0,
                "elo_pre": round(l_overall_pre, 6),
                "elo_post": round(l_overall_post, 6),
                "surface_elo_pre": round(l_surface_pre, 6),
                "surface_elo_post": round(l_surface_post, 6),
            }
        )

        processed_matches += 1

    new_df = pd.DataFrame(rows)
    if incremental and not existing_df.empty and not new_df.empty:
        existing_df["match_date"] = pd.to_datetime(existing_df["match_date"], errors="coerce")
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.sort_values(["match_date", "match_key", "result"], na_position="last")
    elif incremental and not existing_df.empty:
        combined = existing_df
    else:
        combined = new_df

    if not combined.empty and pd.api.types.is_datetime64_any_dtype(combined["match_date"]):
        combined["match_date"] = combined["match_date"].dt.strftime(DATE_FMT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # Snapshot table with latest ratings for inference.
    snapshot_rows = []
    for player_id, elo in overall_ratings.items():
        snapshot_rows.append(
            {
                "player_id": player_id,
                "surface": "ALL",
                "elo": round(elo, 6),
            }
        )
    for (player_id, surface_name), elo in surface_ratings.items():
        if surface_name in SURFACE_LIST:
            snapshot_rows.append(
                {
                    "player_id": player_id,
                    "surface": surface_name,
                    "elo": round(elo, 6),
                }
            )

    snapshot_path = PROCESSED_DIR / f"{tour}_elo_snapshot.csv"
    snapshot_df = pd.DataFrame(snapshot_rows)
    snapshot_df.to_csv(snapshot_path, index=False)
    sync_elo_ratings(combined, tour=tour, snapshot_df=snapshot_df)

    return {
        "tour": tour,
        "output": str(output_path),
        "rows": int(len(combined)),
        "processed_matches": int(processed_matches),
        "snapshot": str(snapshot_path),
    }


def run_elo(incremental: bool = True) -> dict[str, Any]:
    results = {
        "atp": compute_elo_for_tour("atp", incremental=incremental),
        "wta": compute_elo_for_tour("wta", incremental=incremental),
    }

    if LAST_UPDATE_FILE.exists():
        text = LAST_UPDATE_FILE.read_text(encoding="utf-8-sig")
        state = json.loads(text) if text.strip() else {}
    else:
        state = {}

    state.update(
        {
            "elo_last_run": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elo": results,
        }
    )
    LAST_UPDATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    return results


def main() -> None:
    results = run_elo(incremental=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
