from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import MAX_DAILY_BETS, MIN_EDGE_THRESHOLD, MODELS_DIR, ODDS_HISTORY_FILE
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


def run_backtest_for_tour(tour: str, start_capital: float = START_CAPITAL) -> dict[str, Any]:
    pred_path = MODELS_DIR / f"test_predictions_{tour}.csv"
    if not pred_path.exists():
        return {
            "tour": tour,
            "status": "missing_predictions",
            "message": f"Prediction file not found: {pred_path}",
        }

    pred = pd.read_csv(pred_path, low_memory=False)
    odds = _load_odds(ODDS_HISTORY_FILE)
    merged = _join_predictions_with_odds(pred, odds)

    if merged.empty:
        return {
            "tour": tour,
            "status": "no_odds_overlap",
            "message": "No overlapping rows between test predictions and odds_history.",
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
        "rows_with_odds": int(len(scored)),
        "results": strat_results,
    }


def run_backtest(tours: tuple[str, ...] = ("atp", "wta"), start_capital: float = START_CAPITAL) -> dict[str, Any]:
    reports = [run_backtest_for_tour(tour, start_capital=start_capital) for tour in tours]

    summary_rows = []
    for rep in reports:
        if rep.get("status") != "ok":
            summary_rows.append(
                {
                    "tour": rep.get("tour"),
                    "strategy": "n/a",
                    "status": rep.get("status"),
                    "message": rep.get("message"),
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
    report = run_backtest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
