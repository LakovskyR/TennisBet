from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    DEFAULT_CAPITAL,
    KELLY_FRACTION,
    MAX_DAILY_BETS,
    MAX_DAILY_CAPITAL_PCT,
    MIN_BET_AMOUNT,
    MIN_EDGE_THRESHOLD,
    ODDS_UPCOMING_FILE,
)
from src.player_aliases import canonicalize_player_name, load_player_aliases, normalize_player_name


logger = logging.getLogger("value_engine")


def _norm_name(name: Any) -> str:
    return normalize_player_name(name)


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    required = ["match_date", "p1_name", "p2_name", "ensemble_prob_p1", "catboost_prob", "xgboost_prob", "confidence_tier", "model_agreement"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _load_odds(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)

    if "player_1_resolved" not in df.columns:
        df["player_1_resolved"] = df.get("player_1")
    if "player_2_resolved" not in df.columns:
        df["player_2_resolved"] = df.get("player_2")

    needed = [
        "match_date",
        "tournament",
        "surface",
        "player_1",
        "player_2",
        "player_1_resolved",
        "player_2_resolved",
        "odds_p1",
        "odds_p2",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = pd.NA

    return df[needed]


def _join_predictions_with_odds(pred: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or odds.empty:
        return pd.DataFrame()

    pred = pred.copy()
    odds = odds.copy()
    aliases = load_player_aliases()

    pred["match_date"] = pd.to_datetime(pred["match_date"], errors="coerce").dt.date.astype("string")
    odds["match_date"] = pd.to_datetime(odds["match_date"], errors="coerce").dt.date.astype("string")

    odds_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in odds.to_dict(orient="records"):
        date_key = str(row.get("match_date"))
        n1 = canonicalize_player_name(row.get("player_1_resolved") or row.get("player_1"), aliases)
        n2 = canonicalize_player_name(row.get("player_2_resolved") or row.get("player_2"), aliases)
        key = (date_key, min(n1, n2), max(n1, n2))
        if key not in odds_map:
            odds_map[key] = row

    joined_rows = []
    unmatched_rows: list[str] = []
    for row in pred.to_dict(orient="records"):
        date_key = str(row.get("match_date"))
        p1 = canonicalize_player_name(row.get("p1_name"), aliases)
        p2 = canonicalize_player_name(row.get("p2_name"), aliases)
        key = (date_key, min(p1, p2), max(p1, p2))

        odds_row = odds_map.get(key)
        if odds_row is None:
            unmatched_rows.append(f"{row.get('match_date')}:{row.get('p1_name')} vs {row.get('p2_name')}")
            continue

        odds_p1_raw = pd.to_numeric(odds_row.get("odds_p1"), errors="coerce")
        odds_p2_raw = pd.to_numeric(odds_row.get("odds_p2"), errors="coerce")

        if p1 == canonicalize_player_name(odds_row.get("player_1_resolved") or odds_row.get("player_1"), aliases):
            odds_p1 = odds_p1_raw
            odds_p2 = odds_p2_raw
        else:
            odds_p1 = odds_p2_raw
            odds_p2 = odds_p1_raw

        if pd.isna(odds_p1) or pd.isna(odds_p2) or odds_p1 <= 1 or odds_p2 <= 1:
            continue

        joined_rows.append(
            {
                **row,
                "tournament": odds_row.get("tournament"),
                "surface": odds_row.get("surface"),
                "odds_p1": float(odds_p1),
                "odds_p2": float(odds_p2),
            }
        )

    if unmatched_rows:
        sample = ", ".join(unmatched_rows[:5])
        logger.warning(
            "Predictions without odds matches: %d/%d. Sample: %s",
            len(unmatched_rows),
            len(pred),
            sample,
        )

    return pd.DataFrame(joined_rows)


def _remove_overround_power(odds_p1: pd.Series, odds_p2: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    implied_p1 = 1.0 / odds_p1
    implied_p2 = 1.0 / odds_p2
    overround = implied_p1 + implied_p2

    fair_p1 = (implied_p1 / overround).astype(float)
    fair_p2 = (implied_p2 / overround).astype(float)

    valid = (
        odds_p1.notna()
        & odds_p2.notna()
        & np.isfinite(odds_p1)
        & np.isfinite(odds_p2)
        & (odds_p1 > 1.0)
        & (odds_p2 > 1.0)
        & np.isfinite(overround)
        & (overround > 1.0)
    )
    if not valid.any():
        return fair_p1, fair_p2, overround

    q1 = implied_p1.loc[valid].to_numpy(dtype=float)
    q2 = implied_p2.loc[valid].to_numpy(dtype=float)

    low = np.ones(len(q1), dtype=float)
    high = np.full(len(q1), 2.0, dtype=float)

    for _ in range(8):
        high_sum = np.power(q1, high) + np.power(q2, high)
        needs_more = high_sum > 1.0
        if not np.any(needs_more):
            break
        high = np.where(needs_more, high * 2.0, high)

    for _ in range(32):
        mid = (low + high) / 2.0
        mid_sum = np.power(q1, mid) + np.power(q2, mid)
        low = np.where(mid_sum > 1.0, mid, low)
        high = np.where(mid_sum > 1.0, high, mid)

    exponent = high
    power_p1 = np.power(q1, exponent)
    power_p2 = np.power(q2, exponent)
    power_total = power_p1 + power_p2
    power_total = np.where(power_total > 0, power_total, 1.0)

    fair_p1.loc[valid] = power_p1 / power_total
    fair_p2.loc[valid] = power_p2 / power_total
    return fair_p1, fair_p2, overround


def _compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ensemble_prob_p1"] = pd.to_numeric(out["ensemble_prob_p1"], errors="coerce")
    out["odds_p1"] = pd.to_numeric(out["odds_p1"], errors="coerce")
    out["odds_p2"] = pd.to_numeric(out["odds_p2"], errors="coerce")

    fair_p1, fair_p2, overround = _remove_overround_power(out["odds_p1"], out["odds_p2"])
    out["overround"] = overround
    out["fair_implied_p1"] = fair_p1
    out["fair_implied_p2"] = fair_p2
    out["fair_odds_p1"] = 1.0 / out["fair_implied_p1"]
    out["fair_odds_p2"] = 1.0 / out["fair_implied_p2"]

    out["edge_p1"] = out["ensemble_prob_p1"] - out["fair_implied_p1"]
    out["edge_p2"] = (1.0 - out["ensemble_prob_p1"]) - out["fair_implied_p2"]

    choose_p1 = out["edge_p1"] >= out["edge_p2"]
    out["bet_side"] = np.where(choose_p1, "P1", "P2")
    out["selected_edge"] = np.where(choose_p1, out["edge_p1"], out["edge_p2"])
    out["selected_odds"] = np.where(choose_p1, out["odds_p1"], out["odds_p2"])
    out["selected_prob"] = np.where(choose_p1, out["ensemble_prob_p1"], 1.0 - out["ensemble_prob_p1"])
    out["expected_value_per_euro"] = out["selected_prob"] * out["selected_odds"] - 1.0
    return out


def allocate_bankroll(value_bets: pd.DataFrame, capital: float) -> pd.DataFrame:
    return _allocate_bankroll_with_overrides(
        value_bets=value_bets,
        capital=capital,
        min_edge_threshold=MIN_EDGE_THRESHOLD,
        max_daily_bets=MAX_DAILY_BETS,
        max_daily_capital_pct=MAX_DAILY_CAPITAL_PCT,
    )


def _allocate_bankroll_with_overrides(
    value_bets: pd.DataFrame,
    capital: float,
    min_edge_threshold: float,
    max_daily_bets: int,
    max_daily_capital_pct: float,
) -> pd.DataFrame:
    if value_bets.empty:
        return value_bets

    out = value_bets.copy()
    out["selected_prob"] = pd.to_numeric(out["selected_prob"], errors="coerce")
    out["selected_odds"] = pd.to_numeric(out["selected_odds"], errors="coerce")

    b = out["selected_odds"] - 1.0
    q = 1.0 - out["selected_prob"]
    full_kelly = np.where(
        b > 0,
        ((b * out["selected_prob"]) - q) / b,
        0.0,
    )
    out["kelly_fraction_full"] = np.clip(full_kelly, 0.0, None)
    out["kelly_fraction"] = out["kelly_fraction_full"] * KELLY_FRACTION

    out = out.sort_values(["kelly_fraction", "selected_edge"], ascending=[False, False]).head(max_daily_bets).copy()
    out["raw_allocation_pct"] = out["kelly_fraction"].clip(lower=0.0, upper=max_daily_capital_pct)

    total_pct = float(out["raw_allocation_pct"].sum())
    if total_pct > max_daily_capital_pct and total_pct > 0:
        scale = max_daily_capital_pct / total_pct
        out["allocation_pct"] = out["raw_allocation_pct"] * scale
    else:
        out["allocation_pct"] = out["raw_allocation_pct"]

    if capital <= 0:
        out["recommended_stake"] = 0.0
        return out

    out["recommended_stake"] = (capital * out["allocation_pct"]).round(2)
    positive_mask = out["allocation_pct"] > 0
    out.loc[positive_mask, "recommended_stake"] = out.loc[positive_mask, "recommended_stake"].clip(lower=MIN_BET_AMOUNT)
    out.loc[~positive_mask, "recommended_stake"] = 0.0

    total_stake = float(out["recommended_stake"].sum())
    max_stake = capital * max_daily_capital_pct
    if total_stake > max_stake and total_stake > 0:
        out["recommended_stake"] = (out["recommended_stake"] * (max_stake / total_stake)).round(2)
        positive_mask = out["recommended_stake"] > 0
        out.loc[positive_mask, "recommended_stake"] = out.loc[positive_mask, "recommended_stake"].clip(lower=MIN_BET_AMOUNT)

    return out


def generate_recommendations(
    predictions_path: Path,
    odds_path: Path = ODDS_UPCOMING_FILE,
    capital: float = DEFAULT_CAPITAL,
    min_edge_threshold: float = MIN_EDGE_THRESHOLD,
    max_daily_bets: int = MAX_DAILY_BETS,
    max_daily_capital_pct: float = MAX_DAILY_CAPITAL_PCT,
) -> dict[str, Any]:
    pred = _load_predictions(predictions_path)
    odds = _load_odds(odds_path)

    merged = _join_predictions_with_odds(pred, odds)
    if merged.empty:
        return {
            "status": "no_matches",
            "message": "No overlapping prediction/odds matches found.",
            "recommendations": pd.DataFrame(),
        }

    scored = _compute_edges(merged)
    value = scored[scored["selected_edge"] >= min_edge_threshold].copy()

    if value.empty:
        closest = scored.sort_values("selected_edge", ascending=False).head(1)
        return {
            "status": "skip",
            "message": "SKIP TODAY - no value bets above threshold.",
            "closest": closest,
            "recommendations": pd.DataFrame(),
        }

    recs = _allocate_bankroll_with_overrides(
        value_bets=value,
        capital=capital,
        min_edge_threshold=min_edge_threshold,
        max_daily_bets=max_daily_bets,
        max_daily_capital_pct=max_daily_capital_pct,
    )

    return {
        "status": "value",
        "message": f"{len(recs)} value bet(s) detected",
        "recommendations": recs,
    }


def main() -> None:
    atp_pred = Path("data/processed/atp_predictions.csv")
    if not atp_pred.exists():
        print(json.dumps({"status": "error", "message": f"Prediction file not found: {atp_pred}"}, indent=2))
        return

    result = generate_recommendations(predictions_path=atp_pred)
    serializable = {
        "status": result["status"],
        "message": result["message"],
        "count": int(len(result.get("recommendations", pd.DataFrame()))),
    }
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
