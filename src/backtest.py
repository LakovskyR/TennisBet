from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import MAX_DAILY_BETS, MIN_EDGE_THRESHOLD, MODELS_DIR, ODDS_HISTORY_FILE, PROCESSED_DIR
from src.predictor import add_prediction_columns
from src.value_engine import _compute_edges, _join_predictions_with_odds, _load_odds, allocate_bankroll


START_CAPITAL = 100.0


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _parse_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date: {value}")
    return parsed.normalize()


def _date_window_label(start_date: str | None, end_date: str | None) -> str:
    start = start_date or "min"
    end = end_date or "max"
    return f"{start}_to_{end}".replace(":", "-")


def _load_feature_rows_for_tour(
    tour: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    feature_path = PROCESSED_DIR / f"{tour}_player_features.csv"
    if not feature_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(feature_path, low_memory=False)
    if df.empty:
        return df

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df[df["match_date"].notna()].copy()

    start_ts = _parse_date(start_date)
    end_ts = _parse_date(end_date)
    if start_ts is not None:
        df = df[df["match_date"] >= start_ts]
    if end_ts is not None:
        df = df[df["match_date"] <= end_ts]

    if "tour" in df.columns:
        df = df[df["tour"].astype("string").str.lower() == tour]

    return df.sort_values(["match_date", "match_key"]).reset_index(drop=True)


def _build_predictions_for_tour(
    tour: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, Path, str | None]:
    feature_rows = _load_feature_rows_for_tour(tour=tour, start_date=start_date, end_date=end_date)
    output_path = MODELS_DIR / f"backtest_predictions_{tour}_{_date_window_label(start_date, end_date)}.csv"
    if feature_rows.empty:
        return pd.DataFrame(), output_path, "No historical feature rows found for the requested date window."

    pred = add_prediction_columns(feature_rows, tour=tour)
    pred["match_date"] = pd.to_datetime(pred["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    keep_cols = [
        "match_key",
        "match_date",
        "tour",
        "p1_id",
        "p2_id",
        "p1_name",
        "p2_name",
        "p1_wins",
        "surface",
        "round",
        "catboost_prob",
        "xgboost_prob",
        "ensemble_prob_p1",
        "confidence_tier",
        "model_agreement",
    ]
    out = pred[[col for col in keep_cols if col in pred.columns]].copy()
    out.to_csv(output_path, index=False)
    return out, output_path, None


def _load_static_predictions_for_tour(tour: str) -> tuple[pd.DataFrame, Path]:
    pred_path = MODELS_DIR / f"test_predictions_{tour}.csv"
    if not pred_path.exists():
        return pd.DataFrame(), pred_path
    pred = pd.read_csv(pred_path, low_memory=False)
    if "ensemble_prob_p1" not in pred.columns and "ensemble_prob" in pred.columns:
        pred["ensemble_prob_p1"] = pred["ensemble_prob"]
    return pred, pred_path


def _simulate_strategy_on_matches(
    matches: pd.DataFrame,
    strategy: str,
    start_capital: float,
) -> dict[str, Any]:
    bankroll = start_capital
    total_staked = 0.0
    bets = []
    equity_curve = [{"match_date": None, "equity": bankroll}]

    rng = np.random.default_rng(42)

    for day, day_df in matches.groupby("match_date", sort=True):
        if strategy == "value_dynamic":
            value = day_df[day_df["selected_edge"] >= MIN_EDGE_THRESHOLD].copy()
            if value.empty:
                equity_curve.append({"match_date": str(day), "equity": bankroll})
                continue
            day_bets = allocate_bankroll(value, capital=bankroll).head(MAX_DAILY_BETS).copy()
            if day_bets.empty:
                equity_curve.append({"match_date": str(day), "equity": bankroll})
                continue
        elif strategy == "flat_favorite":
            day_bets = day_df.copy()
            day_bets["bet_side"] = np.where(day_bets["odds_p1"] <= day_bets["odds_p2"], "P1", "P2")
            day_bets["selected_odds"] = np.where(day_bets["bet_side"] == "P1", day_bets["odds_p1"], day_bets["odds_p2"])
            day_bets["recommended_stake"] = 1.0
        elif strategy == "flat_random":
            day_bets = day_df.copy()
            random_side = rng.choice(["P1", "P2"], size=len(day_bets))
            day_bets["bet_side"] = random_side
            day_bets["selected_odds"] = np.where(day_bets["bet_side"] == "P1", day_bets["odds_p1"], day_bets["odds_p2"])
            day_bets["recommended_stake"] = 1.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for row in day_bets.to_dict(orient="records"):
            stake = float(row.get("recommended_stake", 0.0))
            if stake <= 0:
                continue
            if bankroll <= 0:
                break
            stake = min(stake, bankroll)

            p1_wins = int(row.get("p1_wins", 0))
            bet_side = row.get("bet_side", "P1")
            selected_odds = float(row.get("selected_odds", 0.0))
            won = (p1_wins == 1 and bet_side == "P1") or (p1_wins == 0 and bet_side == "P2")

            pnl = stake * (selected_odds - 1.0) if won else -stake
            bankroll += pnl
            total_staked += stake

            bets.append(
                {
                    "match_date": row.get("match_date"),
                    "strategy": strategy,
                    "p1_name": row.get("p1_name"),
                    "p2_name": row.get("p2_name"),
                    "bet_side": bet_side,
                    "odds": selected_odds,
                    "stake": round(stake, 2),
                    "won": int(won),
                    "pnl": round(pnl, 4),
                    "bankroll_after": round(bankroll, 4),
                }
            )

        equity_curve.append({"match_date": str(day), "equity": bankroll})

    bets_df = pd.DataFrame(bets)
    wins = int(bets_df["won"].sum()) if not bets_df.empty else 0
    total_bets = int(len(bets_df))
    win_rate = (wins / total_bets) if total_bets else 0.0

    roi_on_staked = ((bankroll - start_capital) / total_staked) if total_staked else 0.0
    roi_on_capital = (bankroll - start_capital) / start_capital if start_capital else 0.0

    return {
        "strategy": strategy,
        "start_capital": start_capital,
        "end_capital": round(bankroll, 4),
        "profit": round(bankroll - start_capital, 4),
        "total_bets": total_bets,
        "win_rate": win_rate,
        "total_staked": round(total_staked, 4),
        "roi_on_staked": roi_on_staked,
        "roi_on_capital": roi_on_capital,
        "max_drawdown": _max_drawdown([x["equity"] for x in equity_curve]),
        "bets": bets_df,
        "equity": pd.DataFrame(equity_curve),
    }


def run_backtest_for_tour(
    tour: str,
    start_capital: float = START_CAPITAL,
    start_date: str | None = None,
    end_date: str | None = None,
    regenerate_predictions: bool = True,
) -> dict[str, Any]:
    if regenerate_predictions:
        pred, pred_path, pred_message = _build_predictions_for_tour(tour=tour, start_date=start_date, end_date=end_date)
        prediction_source = "regenerated_features"
    else:
        pred, pred_path = _load_static_predictions_for_tour(tour=tour)
        pred_message = None
        prediction_source = "static_test_predictions"

    if pred.empty:
        return {
            "tour": tour,
            "status": "missing_predictions",
            "message": pred_message or f"Prediction file not found: {pred_path}",
            "prediction_source": prediction_source,
            "prediction_file": str(pred_path),
            "backtest_start_date": start_date,
            "backtest_end_date": end_date,
        }

    odds = _load_odds(ODDS_HISTORY_FILE)
    if not odds.empty and "tour" in odds.columns:
        odds = odds[odds["tour"].astype("string").str.lower() == tour].copy()

    odds["match_date"] = pd.to_datetime(odds["match_date"], errors="coerce")
    start_ts = _parse_date(start_date)
    end_ts = _parse_date(end_date)
    if start_ts is not None:
        odds = odds[odds["match_date"] >= start_ts]
    if end_ts is not None:
        odds = odds[odds["match_date"] <= end_ts]

    merged = _join_predictions_with_odds(pred, odds)

    if merged.empty:
        return {
            "tour": tour,
            "status": "no_odds_overlap",
            "message": "No overlapping rows between predictions and odds_history.",
            "prediction_source": prediction_source,
            "prediction_file": str(pred_path),
            "backtest_start_date": start_date,
            "backtest_end_date": end_date,
        }

    scored = _compute_edges(merged)

    strat_results = []
    for strategy in ("value_dynamic", "flat_favorite", "flat_random"):
        strat = _simulate_strategy_on_matches(scored, strategy=strategy, start_capital=start_capital)

        bets_path = MODELS_DIR / f"backtest_bets_{tour}_{strategy}.csv"
        equity_path = MODELS_DIR / f"backtest_equity_{tour}_{strategy}.csv"
        strat["bets"].to_csv(bets_path, index=False)
        strat["equity"].to_csv(equity_path, index=False)

        strat_results.append(
            {
                "tour": tour,
                "strategy": strategy,
                "prediction_source": prediction_source,
                "backtest_start_date": start_date,
                "backtest_end_date": end_date,
                "start_capital": strat["start_capital"],
                "end_capital": strat["end_capital"],
                "profit": strat["profit"],
                "total_bets": strat["total_bets"],
                "win_rate": strat["win_rate"],
                "total_staked": strat["total_staked"],
                "roi_on_staked": strat["roi_on_staked"],
                "roi_on_capital": strat["roi_on_capital"],
                "max_drawdown": strat["max_drawdown"],
                "bets_file": str(bets_path),
                "equity_file": str(equity_path),
            }
        )

    return {
        "tour": tour,
        "status": "ok",
        "prediction_source": prediction_source,
        "prediction_file": str(pred_path),
        "backtest_start_date": start_date,
        "backtest_end_date": end_date,
        "rows_requested": int(len(pred)),
        "rows_with_odds": int(len(scored)),
        "results": strat_results,
    }


def run_backtest(
    tours: tuple[str, ...] = ("atp", "wta"),
    start_capital: float = START_CAPITAL,
    start_date: str | None = None,
    end_date: str | None = None,
    regenerate_predictions: bool = True,
) -> dict[str, Any]:
    reports = [
        run_backtest_for_tour(
            tour,
            start_capital=start_capital,
            start_date=start_date,
            end_date=end_date,
            regenerate_predictions=regenerate_predictions,
        )
        for tour in tours
    ]

    summary_rows = []
    for rep in reports:
        if rep.get("status") != "ok":
            summary_rows.append(
                {
                    "tour": rep.get("tour"),
                    "strategy": "n/a",
                    "status": rep.get("status"),
                    "message": rep.get("message"),
                    "prediction_source": rep.get("prediction_source"),
                    "prediction_file": rep.get("prediction_file"),
                    "backtest_start_date": rep.get("backtest_start_date"),
                    "backtest_end_date": rep.get("backtest_end_date"),
                }
            )
            continue
        summary_rows.extend(rep["results"])

    summary_df = pd.DataFrame(summary_rows)
    summary_path = MODELS_DIR / "backtest_results.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_file": str(summary_path),
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run betting strategy backtests on historical tennis data.")
    parser.add_argument("--tours", nargs="+", choices=["atp", "wta"], default=["atp", "wta"], help="Tours to backtest")
    parser.add_argument("--start-capital", type=float, default=START_CAPITAL, help="Starting bankroll per tour")
    parser.add_argument("--start-date", default=None, help="Inclusive backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Inclusive backtest end date (YYYY-MM-DD)")
    parser.add_argument(
        "--static-predictions",
        action="store_true",
        help="Use saved test_predictions artifacts instead of regenerating predictions from historical feature rows",
    )
    args = parser.parse_args()

    report = run_backtest(
        tours=tuple(args.tours),
        start_capital=args.start_capital,
        start_date=args.start_date,
        end_date=args.end_date,
        regenerate_predictions=not args.static_predictions,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
