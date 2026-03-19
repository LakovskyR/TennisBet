from __future__ import annotations

import argparse
import html
import json
import logging
import os
import smtplib
from datetime import UTC, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import (
    APP_LOG_FILE,
    BANKROLL_LOG_FILE,
    DEFAULT_CAPITAL,
    KELLY_FRACTION,
    LAST_UPDATE_FILE,
    MAX_DAILY_BETS,
    MIN_EDGE_THRESHOLD,
    MODELS_DIR,
    ODDS_HISTORY_FILE,
    ODDS_UPCOMING_FILE,
    PREDICTION_LOG_FILE,
    PROCESSED_DIR,
)
from src.data_pipeline import run_pipeline
from src.data_updater import get_staleness_status, update_data_sources
from src.elo_engine import run_elo
from src.model_training import maybe_retrain_models
from src.odds_scraper import refresh_odds
from src.performance_report import generate_performance_report
from src.predictor import predict_from_odds
from src.sqlite_storage import (
    initialize_database,
    load_bankroll_state_payload,
    load_prediction_log_frame,
    sync_bankroll_state,
    sync_odds_frame,
    sync_player_aliases,
    sync_prediction_log_frame,
    sync_reference_players,
)
from src.value_engine import generate_recommendations

PREDICTION_LOG_COLUMNS = [
    "prediction_id",
    "created_at",
    "match_date",
    "tour",
    "match",
    "p1_name",
    "p2_name",
    "bet_side",
    "probability",
    "odds",
    "edge",
    "stake",
    "confidence_tier",
    "model_agreement",
    "result",
    "pnl",
    "status",
    "resolved_at",
]

REPORT_HTML_FILE = LAST_UPDATE_FILE.parent / "daily_report_latest.html"
PREDICTION_FILE_COLUMNS = [
    "match_date",
    "match_key",
    "tour",
    "p1_name",
    "p2_name",
    "ensemble_prob_p1",
    "catboost_prob",
    "xgboost_prob",
    "confidence_tier",
    "model_agreement",
]


def _load_dotenv_if_present(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_gmail_secrets_if_present(path: Path = Path("GMAIL_secrets.txt")) -> None:
    if not path.exists():
        alt = Path("gmail_secrets.txt")
        if not alt.exists():
            return
        path = alt

    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or "→" not in line:
            continue
        key, value = line.split("→", 1)
        data[key.strip()] = value.strip()

    gmail_address = data.get("GMAIL_ADDRESS", "")
    app_password = data.get("GMAIL_APP_PASSWORD", "").replace(" ", "")

    if gmail_address:
        os.environ.setdefault("EMAIL_FROM", gmail_address)
        os.environ.setdefault("EMAIL_TO", gmail_address)
    if app_password:
        os.environ.setdefault("EMAIL_PASSWORD", app_password)
    os.environ.setdefault("SMTP_HOST", "smtp.gmail.com")
    os.environ.setdefault("SMTP_PORT", "587")


def _bootstrap_files() -> None:
    initialize_database()
    sync_reference_players()
    sync_player_aliases()
    PREDICTION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not PREDICTION_LOG_FILE.exists():
        pd.DataFrame(columns=PREDICTION_LOG_COLUMNS).to_csv(PREDICTION_LOG_FILE, index=False)
    if not BANKROLL_LOG_FILE.exists():
        BANKROLL_LOG_FILE.write_text(
            json.dumps(
                {
                    "capital": DEFAULT_CAPITAL,
                    "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "history": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    sync_prediction_log_frame(_safe_read_csv(PREDICTION_LOG_FILE, PREDICTION_LOG_COLUMNS))
    sync_odds_frame(_safe_read_csv(ODDS_HISTORY_FILE), mark_current_upcoming=False, source_name="odds_history")
    sync_odds_frame(_safe_read_csv(ODDS_UPCOMING_FILE), mark_current_upcoming=True, source_name="odds_upcoming")
    sync_bankroll_state(_load_bankroll_state())


def _download_raw_reference_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def _ensure_player_reference_files() -> dict[str, Any]:
    downloaded: list[str] = []
    failures: list[str] = []
    targets = {
        "atp": PROCESSED_DIR.parent / "raw" / "tennis_atp" / "atp_players.csv",
        "wta": PROCESSED_DIR.parent / "raw" / "tennis_wta" / "wta_players.csv",
    }

    for tour, path in targets.items():
        if path.exists() and path.stat().st_size > 0:
            continue
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{path.name}"
        try:
            _download_raw_reference_file(url, path)
            downloaded.append(str(path))
        except Exception as exc:
            failures.append(f"{tour}: {exc}")

    return {"downloaded": downloaded, "failures": failures}


def _safe_read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=columns or [])
    if columns is None:
        return df
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def _load_bankroll_state() -> dict[str, Any]:
    state = load_bankroll_state_payload(fallback_to_file=False)
    if "capital" not in state:
        state["capital"] = float(DEFAULT_CAPITAL)
    if "history" not in state or not isinstance(state["history"], list):
        state["history"] = []
    return state


def _load_last_update() -> dict[str, Any]:
    if not LAST_UPDATE_FILE.exists():
        return {}
    try:
        text = LAST_UPDATE_FILE.read_text(encoding="utf-8-sig")
        return json.loads(text) if text.strip() else {}
    except Exception:
        return {}


def _resolve_staleness(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    last_new_match = state.get("last_new_match")
    if isinstance(last_new_match, str) and last_new_match:
        try:
            last_new_date = datetime.strptime(last_new_match, "%Y-%m-%d").date()
            return get_staleness_status(last_new_date), last_new_match
        except Exception:
            pass
    return state.get("staleness", {}), str(last_new_match or "unknown")


def _load_prediction_log() -> pd.DataFrame:
    df = load_prediction_log_frame(fallback_to_csv=False)
    for col in PREDICTION_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[PREDICTION_LOG_COLUMNS]


def _save_prediction_log(df: pd.DataFrame) -> None:
    PREDICTION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    for col in PREDICTION_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df[PREDICTION_LOG_COLUMNS].to_csv(PREDICTION_LOG_FILE, index=False)
    sync_prediction_log_frame(df[PREDICTION_LOG_COLUMNS].copy())


def _prediction_id(row: pd.Series | dict[str, Any], tour: str) -> str:
    match_date = str(row.get("match_date", "unknown"))
    p1 = str(row.get("p1_name", "p1"))
    p2 = str(row.get("p2_name", "p2"))
    side = str(row.get("bet_side", "P1"))
    return f"{tour}:{match_date}:{p1}:{p2}:{side}".lower()


def _refresh_predictions(tours: tuple[str, ...]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for tour in tours:
        output_path = PROCESSED_DIR / f"{tour}_predictions.csv"
        pred = predict_from_odds(tour=tour, output_path=output_path)
        if pred.empty:
            pd.DataFrame(columns=PREDICTION_FILE_COLUMNS).to_csv(output_path, index=False)
        results[tour] = {
            "rows": int(len(pred)),
            "output_file": str(output_path),
        }
    return results


def _collect_recommendations(tours: tuple[str, ...], capital: float) -> dict[str, Any]:
    by_tour: dict[str, Any] = {}
    combined: list[pd.DataFrame] = []
    closest_rows: list[pd.DataFrame] = []
    analyzed_matches = 0

    for tour in tours:
        pred_path = PROCESSED_DIR / f"{tour}_predictions.csv"
        pred_df = _safe_read_csv(pred_path)
        analyzed_matches += int(len(pred_df))

        result = generate_recommendations(
            predictions_path=pred_path,
            odds_path=ODDS_UPCOMING_FILE,
            capital=capital,
            min_edge_threshold=MIN_EDGE_THRESHOLD,
            max_daily_bets=MAX_DAILY_BETS,
        )
        by_tour[tour] = result

        if result.get("status") == "value":
            recs = result.get("recommendations", pd.DataFrame()).copy()
            if not recs.empty:
                recs["tour"] = tour
                combined.append(recs)
        elif result.get("status") == "skip":
            closest = result.get("closest", pd.DataFrame())
            if isinstance(closest, pd.DataFrame) and not closest.empty:
                closest = closest.copy()
                closest["tour"] = tour
                closest_rows.append(closest)

    combined_df = (
        pd.concat(combined, ignore_index=True).sort_values("selected_edge", ascending=False)
        if combined
        else pd.DataFrame()
    )
    closest_df = (
        pd.concat(closest_rows, ignore_index=True).sort_values("selected_edge", ascending=False)
        if closest_rows
        else pd.DataFrame()
    )
    total_stake = float(pd.to_numeric(combined_df.get("recommended_stake"), errors="coerce").fillna(0.0).sum()) if not combined_df.empty else 0.0

    return {
        "by_tour": by_tour,
        "recommendations": combined_df,
        "closest": closest_df,
        "analyzed_matches": analyzed_matches,
        "total_stake": total_stake,
    }


def _log_recommendations(recommendations: pd.DataFrame) -> int:
    if recommendations.empty:
        return 0

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_df = _load_prediction_log()
    existing_ids = set(log_df["prediction_id"].astype("string").fillna("").tolist())
    new_rows: list[dict[str, Any]] = []

    for _, row in recommendations.iterrows():
        tour = str(row.get("tour", "")).lower()
        pid = _prediction_id(row, tour)
        if pid in existing_ids:
            continue
        new_rows.append(
            {
                "prediction_id": pid,
                "created_at": now,
                "match_date": row.get("match_date"),
                "tour": tour,
                "match": f"{row.get('p1_name')} vs {row.get('p2_name')}",
                "p1_name": row.get("p1_name"),
                "p2_name": row.get("p2_name"),
                "bet_side": row.get("bet_side"),
                "probability": row.get("selected_prob"),
                "odds": row.get("selected_odds"),
                "edge": row.get("selected_edge"),
                "stake": row.get("recommended_stake"),
                "confidence_tier": row.get("confidence_tier"),
                "model_agreement": row.get("model_agreement"),
                "result": pd.NA,
                "pnl": pd.NA,
                "status": "open",
                "resolved_at": pd.NA,
            }
        )

    if not new_rows:
        return 0
    updated = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)
    _save_prediction_log(updated)
    return len(new_rows)


def _subject_for_report(report: dict[str, Any]) -> str:
    report_date = report["generated_at"][:10]
    recs = report["recommendations"]
    errors = report["errors"]
    if errors:
        return f"TennisBet Daily - {report_date} - ERROR"
    if recs.empty:
        return f"TennisBet Daily - {report_date} - SKIP TODAY"
    return f"TennisBet Daily - {report_date} - {len(recs)} bet(s)"


def _html_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "<p>No rows.</p>"

    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                if "edge" in col or "prob" in col or "agreement" in col:
                    text = f"{value:.2%}"
                elif "odds" in col:
                    text = f"{value:.2f}"
                elif "stake" in col:
                    text = f"EUR {value:.2f}"
                else:
                    text = f"{value:.4f}"
            else:
                text = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _display_model_name(model_name: str) -> str:
    aliases = {
        "catboost": "CatBoost",
        "xgboost": "XGBoost",
        "lgbm": "LightGBM",
        "rf": "RandomForest",
        "elasticnet": "ElasticNet",
        "logreg": "LogReg",
    }
    return aliases.get(str(model_name).lower(), str(model_name))


def _format_model_type(report_payload: dict[str, Any]) -> str:
    ensemble_config = report_payload.get("ensemble_config", {})
    weights = ensemble_config.get("weights", {})
    if not isinstance(weights, dict) or not weights:
        return "Unavailable"
    ordered = sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
    labels = "/".join(_display_model_name(name) for name, _ in ordered)
    pct = "/".join(f"{float(weight) * 100:.0f}%" for _, weight in ordered)
    if len(ordered) == 1:
        return labels
    return f"{labels} ensemble ({pct} blend)"


def _load_model_info(tours: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for tour in tours:
        report_path = MODELS_DIR / f"model_report_{tour}.json"
        if not report_path.exists():
            continue
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        metrics = payload.get("metrics", {}).get("ensemble", {})
        split_summary = payload.get("split_summary", {})
        train_summary = split_summary.get("train", {})
        test_summary = split_summary.get("test", {})
        trained_at = str(payload.get("trained_at", ""))[:10] or "unknown"
        rows.append(
            {
                "tour": str(tour).upper(),
                "model_type": _format_model_type(payload),
                "test_accuracy": f"{float(metrics.get('accuracy', 0.0)):.2%}",
                "log_loss": f"{float(metrics.get('log_loss', 0.0)):.3f}",
                "ece": f"{float(metrics.get('ece', 0.0)):.3f}",
                "last_trained": trained_at,
                "training_window": (
                    f"{train_summary.get('date_min', 'n/a')} → {train_summary.get('date_max', 'n/a')} "
                    f"({int(payload.get('rows_train', 0)):,} rows)"
                ),
                "test_window": (
                    f"{test_summary.get('date_min', 'n/a')} → {test_summary.get('date_max', 'n/a')} "
                    f"({int(payload.get('rows_test', 0)):,} rows)"
                ),
            }
        )
    return pd.DataFrame(rows)


def _compute_30day_forecast(capital: float, lookback_days: int = 90) -> dict[str, Any]:
    log_df = _load_prediction_log().copy()
    if log_df.empty:
        return {
            "status": "insufficient_history",
            "message": "Insufficient betting history for 30-day forecast. Need 30+ resolved bets.",
            "lookback_days": lookback_days,
        }

    log_df["resolved_dt"] = pd.to_datetime(log_df.get("resolved_at"), errors="coerce", utc=True)
    log_df["match_dt"] = pd.to_datetime(log_df.get("match_date"), errors="coerce", utc=True)
    status = log_df.get("status", pd.Series("", index=log_df.index)).astype(str).str.lower()
    result = log_df.get("result", pd.Series("", index=log_df.index)).astype(str).str.lower()
    log_df["analysis_dt"] = log_df["resolved_dt"].where(log_df["resolved_dt"].notna(), log_df["match_dt"])
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    resolved_mask = status.eq("resolved") | result.isin(["win", "loss", "push"])
    recent = log_df[resolved_mask & (log_df["analysis_dt"] >= cutoff)].copy()
    if len(recent) < 30:
        return {
            "status": "insufficient_history",
            "message": "Insufficient betting history for 30-day forecast. Need 30+ resolved bets.",
            "lookback_days": lookback_days,
            "resolved_bets": int(len(recent)),
        }

    recent["edge"] = pd.to_numeric(recent.get("edge"), errors="coerce")
    recent = recent[recent["edge"].notna()].copy()
    if len(recent) < 30:
        return {
            "status": "insufficient_history",
            "message": "Insufficient betting history for 30-day forecast. Need 30+ resolved bets.",
            "lookback_days": lookback_days,
            "resolved_bets": int(len(recent)),
        }

    bet_days = recent["analysis_dt"].dt.date.nunique()
    avg_bets_per_day = float(len(recent) / max(bet_days, 1))
    avg_edge = float(recent["edge"].mean())
    edge_std = float(recent["edge"].std(ddof=1)) if len(recent) > 1 else 0.0
    edge_se = edge_std / max(len(recent) ** 0.5, 1.0)
    edge_ci = 1.96 * edge_se

    win_mask = result.eq("win")
    loss_mask = result.eq("loss")
    win_loss = recent[win_mask | loss_mask].copy()
    hit_rate = float(win_mask.loc[win_loss.index].mean()) if not win_loss.empty else 0.0

    def _project(edge_value: float) -> tuple[float, float]:
        daily_return = avg_bets_per_day * KELLY_FRACTION * edge_value
        daily_return = max(daily_return, -0.99)
        multiplier = float((1.0 + daily_return) ** 30)
        return daily_return, multiplier

    conservative_daily_return, conservative_multiplier = _project(avg_edge - edge_ci)
    expected_daily_return, expected_multiplier = _project(avg_edge)
    optimistic_daily_return, optimistic_multiplier = _project(avg_edge + edge_ci)

    return {
        "status": "ok",
        "lookback_days": lookback_days,
        "resolved_bets": int(len(recent)),
        "hit_rate": hit_rate,
        "avg_edge": avg_edge,
        "avg_bets_per_day": avg_bets_per_day,
        "kelly_fraction": float(KELLY_FRACTION),
        "capital_start": float(capital),
        "conservative_daily_return": conservative_daily_return,
        "expected_daily_return": expected_daily_return,
        "optimistic_daily_return": optimistic_daily_return,
        "conservative_multiplier": conservative_multiplier,
        "expected_multiplier": expected_multiplier,
        "optimistic_multiplier": optimistic_multiplier,
        "conservative_capital": float(capital) * conservative_multiplier,
        "expected_capital": float(capital) * expected_multiplier,
        "optimistic_capital": float(capital) * optimistic_multiplier,
    }


def _build_html_report(report: dict[str, Any]) -> str:
    recs: pd.DataFrame = report["recommendations"]
    closest: pd.DataFrame = report["closest"]
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]
    state = report["state"]
    bankroll = report["bankroll"]
    odds_report = report["steps"].get("odds_refresh", {})
    performance_report = report.get("performance_report", {})
    model_info: pd.DataFrame = report.get("model_info", pd.DataFrame())
    forecast_30day: dict[str, Any] = report.get("forecast_30day", {})

    recommendation_block = ""
    if recs.empty:
        recommendation_block = "<h2>SKIP TODAY</h2><p>No value bets detected.</p>"
        if not closest.empty:
            top = closest.iloc[0]
            recommendation_block += (
                "<p>Closest to threshold: "
                f"{html.escape(str(top.get('p1_name')))} vs {html.escape(str(top.get('p2_name')))} "
                f"at {float(top.get('selected_edge', 0.0)):.2%} edge."
                "</p>"
            )
    else:
        recommendation_block = (
            f"<h2>{len(recs)} value bet(s) detected</h2>"
            f"<p>Suggested allocation: EUR {report['total_stake']:.2f}</p>"
            + _html_table(
                recs,
                [
                    "tour",
                    "match_date",
                    "p1_name",
                    "p2_name",
                    "bet_side",
                    "selected_prob",
                    "selected_odds",
                    "selected_edge",
                    "recommended_stake",
                    "confidence_tier",
                    "model_agreement",
                ],
            )
        )

    error_block = ""
    if errors:
        items = "".join(f"<li>{html.escape(msg)}</li>" for msg in errors)
        error_block = f"<h2>Errors</h2><ul>{items}</ul>"

    warning_block = ""
    if warnings:
        items = "".join(f"<li>{html.escape(msg)}</li>" for msg in warnings)
        warning_block = f"<h2>Warnings</h2><ul>{items}</ul>"

    performance_block = ""
    if performance_report:
        output_path = performance_report.get("output_path", "")
        summary_text = performance_report.get("summary_text", "")
        performance_block = (
            "<h2>Weekly Performance Dashboard</h2>"
            f"<p><strong>Status:</strong> {html.escape(str(performance_report.get('status', 'unknown')))}</p>"
            f"<p><strong>HTML path:</strong> {html.escape(str(output_path))}</p>"
            f"<pre>{html.escape(str(summary_text))}</pre>"
        )

    model_info_block = ""
    if not model_info.empty:
        model_info_block = (
            "<h2>Model Info</h2>"
            + _html_table(
                model_info,
                [
                    "tour",
                    "model_type",
                    "test_accuracy",
                    "log_loss",
                    "ece",
                    "last_trained",
                    "training_window",
                    "test_window",
                ],
            )
        )

    forecast_block = ""
    if forecast_30day:
        if forecast_30day.get("status") != "ok":
            forecast_block = (
                "<h2>30-Day Strategy Outlook</h2>"
                f"<p>{html.escape(str(forecast_30day.get('message', 'Forecast unavailable.')))}</p>"
            )
        else:
            forecast_block = (
                f"<h2>30-Day Strategy Outlook (based on last {int(forecast_30day.get('lookback_days', 90))} days)</h2>"
                "<ul>"
                f"<li>Resolved bets: {int(forecast_30day.get('resolved_bets', 0))} | "
                f"Hit rate: {float(forecast_30day.get('hit_rate', 0.0)):.1%} | "
                f"Avg edge: {float(forecast_30day.get('avg_edge', 0.0)):.1%} | "
                f"Avg bets/day: {float(forecast_30day.get('avg_bets_per_day', 0.0)):.2f}</li>"
                f"<li>Conservative estimate: EUR {float(forecast_30day.get('capital_start', 0.0)):.2f} → "
                f"EUR {float(forecast_30day.get('conservative_capital', 0.0)):.2f} "
                f"(×{float(forecast_30day.get('conservative_multiplier', 0.0)):.2f})</li>"
                f"<li>Expected estimate: EUR {float(forecast_30day.get('capital_start', 0.0)):.2f} → "
                f"EUR {float(forecast_30day.get('expected_capital', 0.0)):.2f} "
                f"(×{float(forecast_30day.get('expected_multiplier', 0.0)):.2f})</li>"
                f"<li>Optimistic estimate: EUR {float(forecast_30day.get('capital_start', 0.0)):.2f} → "
                f"EUR {float(forecast_30day.get('optimistic_capital', 0.0)):.2f} "
                f"(×{float(forecast_30day.get('optimistic_multiplier', 0.0)):.2f})</li>"
                "</ul>"
            )

    staleness, last_new_match = _resolve_staleness(state)
    odds_message = odds_report.get("message", "Odds refresh did not run.")

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TennisBet Daily</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #1f2937; margin: 24px; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ background: #f3f4f6; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f9fafb; }}
    ul {{ padding-left: 20px; }}
  </style>
</head>
<body>
  <h1>TennisBet Daily Report</h1>
  <div class="meta">
    <p><strong>Generated:</strong> {html.escape(report["generated_at"])}</p>
    <p><strong>Capital:</strong> EUR {float(bankroll.get("capital", DEFAULT_CAPITAL)):.2f}</p>
    <p><strong>Data freshness:</strong> {html.escape(str(staleness.get("message", "unknown")))}</p>
    <p><strong>Last new match:</strong> {html.escape(str(last_new_match))}</p>
    <p><strong>Analyzed matches:</strong> {int(report["analyzed_matches"])}</p>
    <p><strong>Odds status:</strong> {html.escape(str(odds_message))}</p>
  </div>
  {performance_block}
  {recommendation_block}
  {model_info_block}
  {forecast_block}
  {warning_block}
  {error_block}
</body>
</html>
""".strip()


def _send_email(subject: str, html_body: str) -> None:
    email_to = os.environ["EMAIL_TO"]
    email_from = os.environ["EMAIL_FROM"]
    email_password = os.environ["EMAIL_PASSWORD"].replace(" ", "")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.attach(MIMEText("HTML report attached in email body.", "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
        server.starttls()
        server.login(email_from, email_password)
        server.sendmail(email_from, [addr.strip() for addr in email_to.split(",") if addr.strip()], msg.as_string())


def run_daily_report(
    tours: tuple[str, ...] = ("atp", "wta"),
    send_email: bool = True,
    bankroll: float | None = None,
    skip_empty: bool = False,
    skip_retrain: bool = False,
) -> dict[str, Any]:
    _load_dotenv_if_present()
    _load_gmail_secrets_if_present()
    _bootstrap_files()

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps": {},
        "errors": [],
        "warnings": [],
    }

    bankroll_state = _load_bankroll_state()
    capital = float(bankroll) if bankroll is not None else float(bankroll_state.get("capital", DEFAULT_CAPITAL))
    report["bankroll"] = {
        **bankroll_state,
        "capital": capital,
        "override_used": bankroll is not None,
    }

    try:
        report["steps"]["data_update"] = update_data_sources()
    except Exception as exc:
        report["errors"].append(f"Data update failed: {exc}")

    player_ref_report = _ensure_player_reference_files()
    report["steps"]["player_reference_files"] = player_ref_report
    for failure in player_ref_report.get("failures", []):
        report["errors"].append(f"Player reference download failed: {failure}")

    try:
        report["steps"]["pipeline"] = run_pipeline(incremental=True)
    except Exception as exc:
        report["errors"].append(f"Pipeline failed: {exc}")

    try:
        report["steps"]["elo"] = run_elo(incremental=True)
    except Exception as exc:
        report["errors"].append(f"ELO update failed: {exc}")

    if skip_retrain:
        report["steps"]["model_retraining"] = {"triggered": False, "skipped": True, "reason": "cli_flag"}
        logging.info("Model retraining skipped (--skip-retrain flag)")
    else:
        try:
            retrain_report = maybe_retrain_models(tours=tours)
            report["steps"]["model_retraining"] = retrain_report
            if retrain_report.get("triggered"):
                report["warnings"].append(
                    f"Models retrained for: {', '.join(retrain_report.get('tours', []))}."
                )
        except Exception as exc:
            report["errors"].append(f"Model retraining failed: {exc}")
            report["steps"]["model_retraining"] = {"triggered": False, "error": str(exc)}

    try:
        odds_report = refresh_odds()
        report["steps"]["odds_refresh"] = odds_report
        if not odds_report.get("success", False):
            report["warnings"].append(str(odds_report.get("message", "Odds refresh fell back to manual mode.")))
    except Exception as exc:
        report["errors"].append(f"Odds refresh failed: {exc}")
        report["steps"]["odds_refresh"] = {"success": False, "message": str(exc)}

    try:
        report["steps"]["predictions"] = _refresh_predictions(tours)
    except Exception as exc:
        report["errors"].append(f"Prediction refresh failed: {exc}")
        report["steps"]["predictions"] = {}

    recommendation_bundle = _collect_recommendations(tours=tours, capital=capital)
    report["recommendations"] = recommendation_bundle["recommendations"]
    report["closest"] = recommendation_bundle["closest"]
    report["analyzed_matches"] = recommendation_bundle["analyzed_matches"]
    report["total_stake"] = recommendation_bundle["total_stake"]
    report["model_info"] = _load_model_info(tours)
    report["forecast_30day"] = _compute_30day_forecast(capital=capital)
    report["steps"]["recommendations"] = {
        "by_tour": {
            tour: {
                "status": rec.get("status"),
                "message": rec.get("message"),
                "count": int(len(rec.get("recommendations", pd.DataFrame())))
                if isinstance(rec.get("recommendations"), pd.DataFrame)
                else 0,
            }
            for tour, rec in recommendation_bundle["by_tour"].items()
        }
    }

    try:
        logged = _log_recommendations(report["recommendations"])
        report["steps"]["prediction_log"] = {"rows_logged": logged}
    except Exception as exc:
        report["errors"].append(f"Prediction log update failed: {exc}")
        report["steps"]["prediction_log"] = {"rows_logged": 0}

    report["state"] = _load_last_update()

    if datetime.today().weekday() == 0:
        try:
            performance_output = PROCESSED_DIR / "performance_report.html"
            perf_report = generate_performance_report(output_path=performance_output, lookback_days=90)
            report["performance_report"] = perf_report
            report["steps"]["performance_report"] = perf_report
            logging.info("Performance report generated at %s", perf_report.get("output_path"))
        except Exception as exc:
            report["errors"].append(f"Performance report generation failed: {exc}")
            report["steps"]["performance_report"] = {"status": "error", "error": str(exc)}

    html_body = _build_html_report(report)
    REPORT_HTML_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_HTML_FILE.write_text(html_body, encoding="utf-8")
    report["html_file"] = str(REPORT_HTML_FILE)

    subject = _subject_for_report(report)
    report["email_subject"] = subject

    if send_email:
        if skip_empty and report["recommendations"].empty and not report["errors"]:
            report["steps"]["email"] = {"sent": False, "reason": "skip_empty_no_recommendations"}
        else:
            required = ["EMAIL_TO", "EMAIL_FROM", "EMAIL_PASSWORD"]
            missing = [key for key in required if not os.environ.get(key)]
            if missing:
                report["errors"].append(f"Missing email configuration: {', '.join(missing)}")
            else:
                try:
                    _send_email(subject, html_body)
                    report["steps"]["email"] = {"sent": True}
                except Exception as exc:
                    report["errors"].append(f"Email send failed: {exc}")
                    report["steps"]["email"] = {"sent": False, "error": str(exc)}
    else:
        report["steps"]["email"] = {"sent": False, "reason": "dry_run"}

    serializable = dict(report)
    serializable["recommendations"] = int(len(report["recommendations"]))
    serializable["closest"] = int(len(report["closest"]))
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the headless TennisBet daily report cycle")
    parser.add_argument("--tours", default="atp,wta", help="Comma-separated tours, default atp,wta")
    parser.add_argument("--bankroll", type=float, default=None, help="Override bankroll for this run")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending email")
    parser.add_argument("--skip-empty", action="store_true", help="Skip email send when no bets are recommended")
    parser.add_argument("--skip-retrain", action="store_true", help="Skip automatic model retraining check")
    args = parser.parse_args()

    APP_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(APP_LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    tours = tuple(t.strip() for t in args.tours.split(",") if t.strip())
    report = run_daily_report(
        tours=tours,
        send_email=not args.dry_run,
        bankroll=args.bankroll,
        skip_empty=args.skip_empty,
        skip_retrain=args.skip_retrain,
    )
    logging.info("Daily report cycle completed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
