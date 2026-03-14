from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    MAX_DAILY_BETS,
    MAX_DAILY_CAPITAL_PCT,
    MIN_BET_AMOUNT,
    MIN_EDGE_THRESHOLD,
    ODDS_UPCOMING_FILE,
)


def _norm_name(name: Any) -> str:
    return " ".join(str(name).lower().replace(".", " ").replace("-", " ").split())


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

    pred["match_date"] = pd.to_datetime(pred["match_date"], errors="coerce").dt.date.astype("string")
    odds["match_date"] = pd.to_datetime(odds["match_date"], errors="coerce").dt.date.astype("string")

    odds_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in odds.to_dict(orient="records"):
        date_key = str(row.get("match_date"))
        n1 = _norm_name(row.get("player_1_resolved") or row.get("player_1"))
        n2 = _norm_name(row.get("player_2_resolved") or row.get("player_2"))
        key = (date_key, min(n1, n2), max(n1, n2))
        if key not in odds_map:
            odds_map[key] = row

    joined_rows = []
    for row in pred.to_dict(orient="records"):
        date_key = str(row.get("match_date"))
        p1 = _norm_name(row.get("p1_name"))
        p2 = _norm_name(row.get("p2_name"))
        key = (date_key, min(p1, p2), max(p1, p2))

        odds_row = odds_map.get(key)
        if odds_row is None:
            continue

        odds_p1_raw = pd.to_numeric(odds_row.get("odds_p1"), errors="coerce")
        odds_p2_raw = pd.to_numeric(odds_row.get("odds_p2"), errors="coerce")

        if p1 == _norm_name(odds_row.get("player_1_resolved") or odds_row.get("player_1")):
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

    return pd.DataFrame(joined_rows)


def _compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ensemble_prob_p1"] = pd.to_numeric(out["ensemble_prob_p1"], errors="coerce")

    implied_p1 = 1.0 / out["odds_p1"]
    implied_p2 = 1.0 / out["odds_p2"]
    total_implied = implied_p1 + implied_p2

    out["fair_implied_p1"] = implied_p1 / total_implied
    out["fair_implied_p2"] = implied_p2 / total_implied

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

    conf_mult = {"HIGH": 1.3, "MEDIUM": 1.0, "LOW": 0.7}

    out = value_bets.copy()
    out = out.sort_values("selected_edge", ascending=False).head(max_daily_bets).copy()

    raw_pct = []
    for edge, conf in out[["selected_edge", "confidence_tier"]].itertuples(index=False):
        base_pct = 0.10
        edge_bonus = max(0.0, float(edge) - min_edge_threshold) * 2.0
        pct = min(base_pct + edge_bonus, 0.40) * conf_mult.get(str(conf), 0.8)
        raw_pct.append(pct)

    out["raw_allocation_pct"] = raw_pct

    n = len(out)
    if n == 1:
        out["raw_allocation_pct"] = out["raw_allocation_pct"].clip(lower=0.20, upper=0.40)
    elif n == 2:
        out["raw_allocation_pct"] = out["raw_allocation_pct"].clip(lower=0.15, upper=0.30)
    elif n >= 3:
        out["raw_allocation_pct"] = out["raw_allocation_pct"].clip(lower=0.10, upper=0.20)

    total_pct = float(out["raw_allocation_pct"].sum())
    if total_pct > max_daily_capital_pct and total_pct > 0:
        scale = max_daily_capital_pct / total_pct
        out["allocation_pct"] = out["raw_allocation_pct"] * scale
    else:
        out["allocation_pct"] = out["raw_allocation_pct"]

    out["recommended_stake"] = (capital * out["allocation_pct"]).round(2)
    out["recommended_stake"] = out["recommended_stake"].clip(lower=MIN_BET_AMOUNT)

    total_stake = float(out["recommended_stake"].sum())
    max_stake = capital * max_daily_capital_pct
    if total_stake > max_stake and total_stake > 0:
        out["recommended_stake"] = (out["recommended_stake"] * (max_stake / total_stake)).round(2)
        out["recommended_stake"] = out["recommended_stake"].clip(lower=MIN_BET_AMOUNT)

    return out


def generate_recommendations(
    predictions_path: Path,
    odds_path: Path = ODDS_UPCOMING_FILE,
    capital: float = 5.0,
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
