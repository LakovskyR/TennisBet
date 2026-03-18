from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    BANKROLL_LOG_FILE,
    CUSTOM_ATP_FILE,
    CUSTOM_WTA_FILE,
    DATE_FMT,
    DEFAULT_CAPITAL,
    LAST_UPDATE_FILE,
    MAX_DAILY_BETS,
    MAX_DAILY_CAPITAL_PCT,
    MIN_EDGE_THRESHOLD,
    MODELS_DIR,
    ODDS_HISTORY_FILE,
    ODDS_UPCOMING_FILE,
    PREDICTION_LOG_FILE,
    PROCESSED_DIR,
    RAW_ATP,
    RAW_WTA,
)
from src.data_pipeline import run_pipeline
from src.data_updater import get_staleness_status, update_data_sources
from src.elo_engine import run_elo
from src.odds_scraper import refresh_odds
from src.predictor import predict_from_feature_file, predict_from_odds
from src.value_engine import generate_recommendations

CUSTOM_SCHEMA = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "loser_id",
    "loser_name",
    "score",
    "best_of",
    "round",
    "source",
]

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

ODDS_UPLOAD_REQUIRED = ["match_date", "player_1", "player_2", "odds_p1", "odds_p2"]
ODDS_UPLOAD_COLUMNS = [
    "match_date",
    "match_time",
    "tour",
    "tournament",
    "surface",
    "player_1",
    "player_2",
    "player_1_resolved",
    "player_2_resolved",
    "player_1_id",
    "player_2_id",
    "player_1_match_score",
    "player_2_match_score",
    "odds_p1",
    "odds_p2",
    "source_url",
    "match_id",
    "captured_at",
]


def _ensure_file(path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    if columns is None:
        path.write_text("", encoding="utf-8")
    else:
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def _bootstrap_files() -> None:
    _ensure_file(CUSTOM_ATP_FILE, CUSTOM_SCHEMA)
    _ensure_file(CUSTOM_WTA_FILE, CUSTOM_SCHEMA)
    _ensure_file(PREDICTION_LOG_FILE, PREDICTION_LOG_COLUMNS)
    _ensure_file(ODDS_UPCOMING_FILE, ODDS_UPLOAD_COLUMNS)
    _ensure_file(ODDS_HISTORY_FILE, ODDS_UPLOAD_COLUMNS)

    if not LAST_UPDATE_FILE.exists():
        LAST_UPDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_UPDATE_FILE.write_text(
            json.dumps(
                {
                    "initialized_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "last_new_match": None,
                    "staleness": {"status": "unknown", "message": "No data update run yet."},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

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


def _format_pct(value: Any, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    try:
        return f"{float(value):.{digits}%}"
    except Exception:
        return "-"


def _format_decimal(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _format_currency(value: Any) -> str:
    try:
        if pd.isna(value):
            return "EUR -"
    except Exception:
        pass
    try:
        return f"EUR {float(value):.2f}"
    except Exception:
        return "EUR -"


def _confidence_badge(value: Any) -> str:
    tier = str(value or "").upper()
    if tier == "HIGH":
        return "HIGH"
    if tier == "MEDIUM":
        return "MEDIUM"
    if tier == "LOW":
        return "LOW"
    return "UNKNOWN"


def _edge_band(edge: Any) -> str:
    try:
        edge_val = float(edge)
    except Exception:
        return "unknown"
    if edge_val >= 0.08:
        return "strong"
    if edge_val >= 0.05:
        return "playable"
    return "below-threshold"


def _tour_list(tour_filter: str) -> list[str]:
    return ["atp", "wta"] if tour_filter == "both" else [tour_filter]


@st.cache_data(show_spinner=False)
def _known_player_ids(tour: str) -> set[str]:
    players_file = RAW_ATP / "atp_players.csv" if tour == "atp" else RAW_WTA / "wta_players.csv"
    if not players_file.exists():
        return set()
    try:
        df = pd.read_csv(players_file, usecols=["player_id"], low_memory=False)
    except Exception:
        return set()

    ids: set[str] = set()
    for value in df["player_id"].tolist():
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        try:
            ids.add(str(int(float(value))))
        except Exception:
            text = str(value).strip()
            if text:
                ids.add(text)
    return ids


def _load_bankroll_state() -> dict[str, Any]:
    if not BANKROLL_LOG_FILE.exists():
        return {"capital": float(DEFAULT_CAPITAL), "updated_at": None, "history": []}
    try:
        text = BANKROLL_LOG_FILE.read_text(encoding="utf-8-sig")
        state = json.loads(text) if text.strip() else {}
    except Exception:
        state = {}

    if "capital" not in state:
        state["capital"] = float(DEFAULT_CAPITAL)
    if "history" not in state or not isinstance(state["history"], list):
        state["history"] = []
    return state


def _save_bankroll_state(state: dict[str, Any]) -> None:
    BANKROLL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    BANKROLL_LOG_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _apply_bankroll_update(pnl: float, prediction_id: str, note: str) -> dict[str, Any]:
    state = _load_bankroll_state()
    before = float(state.get("capital", DEFAULT_CAPITAL))
    after = max(0.0, before + float(pnl))

    entry = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prediction_id": prediction_id,
        "note": note,
        "pnl": round(float(pnl), 2),
        "capital_before": round(before, 2),
        "capital_after": round(after, 2),
    }
    history = state.get("history", [])
    if not isinstance(history, list):
        history = []
    history.append(entry)

    state["capital"] = round(after, 2)
    state["updated_at"] = entry["timestamp"]
    state["history"] = history[-5000:]
    _save_bankroll_state(state)
    return entry


def _load_prediction_log() -> pd.DataFrame:
    return _safe_read_csv(PREDICTION_LOG_FILE, PREDICTION_LOG_COLUMNS)


def _save_prediction_log(df: pd.DataFrame) -> None:
    PREDICTION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    for col in PREDICTION_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df[PREDICTION_LOG_COLUMNS].to_csv(PREDICTION_LOG_FILE, index=False)


def _prediction_id(row: pd.Series | dict[str, Any], tour: str) -> str:
    match_date = str(row.get("match_date", "unknown"))
    p1 = str(row.get("p1_name", "p1"))
    p2 = str(row.get("p2_name", "p2"))
    side = str(row.get("bet_side", "P1"))
    return f"{tour}:{match_date}:{p1}:{p2}:{side}".lower()


def _load_last_update() -> dict[str, Any]:
    if not LAST_UPDATE_FILE.exists():
        return {}
    try:
        text = LAST_UPDATE_FILE.read_text(encoding="utf-8-sig")
        return json.loads(text) if text.strip() else {}
    except Exception:
        return {}


def _show_staleness_banner(state: dict[str, Any]) -> None:
    staleness = state.get("staleness", {})
    status = staleness.get("status", "unknown")
    message = staleness.get("message", "No staleness information yet.")

    if status == "fresh":
        st.success(message)
    elif status in {"warning", "stale"}:
        st.warning(message)
    elif status == "critical":
        st.error(message)
    else:
        st.info(message)


def _append_custom_match(tour: str, payload: dict[str, Any]) -> None:
    path = CUSTOM_ATP_FILE if tour == "atp" else CUSTOM_WTA_FILE
    _ensure_file(path, CUSTOM_SCHEMA)

    row = {key: payload.get(key) for key in CUSTOM_SCHEMA}
    row["source"] = "custom"

    current = _safe_read_csv(path, CUSTOM_SCHEMA)
    updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(path, index=False)


def _ingest_uploaded_custom_csv(tour: str, uploaded_file: Any) -> tuple[bool, str]:
    path = CUSTOM_ATP_FILE if tour == "atp" else CUSTOM_WTA_FILE
    _ensure_file(path, CUSTOM_SCHEMA)

    try:
        incoming = pd.read_csv(uploaded_file)
    except Exception as exc:
        return False, f"Could not read CSV: {exc}"

    for col in CUSTOM_SCHEMA:
        if col not in incoming.columns:
            incoming[col] = pd.NA

    incoming = incoming[CUSTOM_SCHEMA]
    incoming["source"] = "custom"

    existing = _safe_read_csv(path, CUSTOM_SCHEMA)
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged.to_csv(path, index=False)
    return True, f"Imported {len(incoming)} custom rows into {path.name}."


def _prepare_uploaded_odds_frame(df: pd.DataFrame, default_tour: str | None) -> tuple[bool, pd.DataFrame | str]:
    missing = [col for col in ODDS_UPLOAD_REQUIRED if col not in df.columns]
    if missing:
        return False, f"Missing required odds columns: {', '.join(missing)}"

    prepared = df.copy()
    for col in ODDS_UPLOAD_COLUMNS:
        if col not in prepared.columns:
            prepared[col] = pd.NA

    if "tour" not in df.columns or prepared["tour"].isna().all():
        if default_tour is None:
            return False, "Odds CSV needs a 'tour' column when the sidebar tour filter is set to both."
        prepared["tour"] = default_tour

    prepared["tour"] = prepared["tour"].astype("string").str.lower().str.strip()
    invalid_tours = ~prepared["tour"].isin(["atp", "wta"])
    if invalid_tours.any():
        return False, "Odds CSV contains invalid tour values. Use 'atp' or 'wta'."

    prepared["player_1_resolved"] = prepared["player_1_resolved"].fillna(prepared["player_1"])
    prepared["player_2_resolved"] = prepared["player_2_resolved"].fillna(prepared["player_2"])
    prepared["match_time"] = prepared["match_time"].fillna("")
    prepared["tournament"] = prepared["tournament"].fillna("Manual upload")
    prepared["surface"] = prepared["surface"].fillna("Unknown")
    prepared["player_1_match_score"] = prepared["player_1_match_score"].fillna(100)
    prepared["player_2_match_score"] = prepared["player_2_match_score"].fillna(100)
    prepared["source_url"] = prepared["source_url"].fillna("manual_upload")
    prepared["captured_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    prepared["odds_p1"] = pd.to_numeric(prepared["odds_p1"], errors="coerce")
    prepared["odds_p2"] = pd.to_numeric(prepared["odds_p2"], errors="coerce")
    prepared = prepared.dropna(subset=["odds_p1", "odds_p2", "match_date", "player_1", "player_2"])
    prepared = prepared[(prepared["odds_p1"] > 1.0) & (prepared["odds_p2"] > 1.0)]
    if prepared.empty:
        return False, "No valid odds rows after validation. Ensure odds are numeric and greater than 1.0."

    prepared["match_date"] = pd.to_datetime(prepared["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    prepared = prepared.dropna(subset=["match_date"])
    if prepared.empty:
        return False, "No valid match_date values in odds CSV."

    if "match_id" not in df.columns or prepared["match_id"].isna().all():
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        prepared["match_id"] = [f"manual-{stamp}-{idx}" for idx in range(len(prepared))]
    prepared["match_id"] = prepared["match_id"].astype("string").fillna("")

    return True, prepared[ODDS_UPLOAD_COLUMNS]


def _ingest_uploaded_odds_csv(uploaded_file: Any, default_tour: str | None) -> tuple[bool, str]:
    try:
        incoming = pd.read_csv(uploaded_file)
    except Exception as exc:
        return False, f"Could not read odds CSV: {exc}"

    ok, prepared = _prepare_uploaded_odds_frame(incoming, default_tour=default_tour)
    if not ok:
        return False, str(prepared)

    assert isinstance(prepared, pd.DataFrame)
    history = _safe_read_csv(ODDS_HISTORY_FILE, ODDS_UPLOAD_COLUMNS)
    prepared.to_csv(ODDS_UPCOMING_FILE, index=False)
    pd.concat([history, prepared], ignore_index=True).to_csv(ODDS_HISTORY_FILE, index=False)
    return True, f"Imported {len(prepared)} odds row(s) into {ODDS_UPCOMING_FILE.name}."


def _run_data_refresh() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    update_report = update_data_sources()
    pipeline_report = run_pipeline(incremental=True)
    elo_report = run_elo(incremental=True)

    state = _load_last_update()
    if state.get("last_new_match"):
        try:
            last_new = datetime.strptime(state["last_new_match"], DATE_FMT).date()
            state["staleness"] = get_staleness_status(last_new)
            LAST_UPDATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass

    return update_report, pipeline_report, elo_report


def _count_matches(tour: str) -> int:
    path = PROCESSED_DIR / f"{tour}_matches_master.csv"
    return len(_safe_read_csv(path))


def _count_custom_matches(tour: str) -> int:
    path = CUSTOM_ATP_FILE if tour == "atp" else CUSTOM_WTA_FILE
    return len(_safe_read_csv(path, CUSTOM_SCHEMA))


def _load_prediction_file(tour: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{tour}_predictions.csv"
    columns = [
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
    return _safe_read_csv(path, columns)


def _count_predicted_matches(tour_filter: str) -> int:
    total = 0
    for tour in _tour_list(tour_filter):
        total += len(_load_prediction_file(tour))
    return total


def _refresh_predictions(tour_filter: str) -> list[str]:
    outputs: list[str] = []
    for tour in _tour_list(tour_filter):
        output_path = PROCESSED_DIR / f"{tour}_predictions.csv"
        pred = predict_from_odds(tour=tour, output_path=output_path)
        if pred.empty:
            input_path = PROCESSED_DIR / f"{tour}_player_features.csv"
            if not input_path.exists():
                continue
            pred = predict_from_feature_file(
                tour=tour,
                input_path=input_path,
                output_path=output_path,
                limit=2000,
            )
            outputs.append(f"{tour.upper()}: {len(pred)} rows (historical fallback)")
        else:
            outputs.append(f"{tour.upper()}: {len(pred)} rows (from upcoming odds)")
    return outputs


def _load_model_metrics() -> pd.DataFrame:
    return _safe_read_csv(MODELS_DIR / "model_metrics.csv")


def _load_backtest_results() -> pd.DataFrame:
    return _safe_read_csv(MODELS_DIR / "backtest_results.csv")


def _load_calibration(tour: str) -> pd.DataFrame:
    return _safe_read_csv(MODELS_DIR / f"calibration_{tour}.csv")


def _load_feature_importance(tour: str) -> pd.DataFrame:
    return _safe_read_csv(MODELS_DIR / f"feature_importance_{tour}.csv")


def _odds_snapshot_status() -> dict[str, Any]:
    if not ODDS_UPCOMING_FILE.exists() or ODDS_UPCOMING_FILE.stat().st_size == 0:
        return {"status": "missing", "message": "No upcoming odds snapshot file."}
    try:
        odds = pd.read_csv(ODDS_UPCOMING_FILE, low_memory=False)
    except Exception as exc:
        return {"status": "error", "message": f"Failed to read odds snapshot: {exc}"}
    if odds.empty:
        return {"status": "empty", "message": "Upcoming odds snapshot is empty."}

    if "captured_at" not in odds.columns:
        return {"status": "unknown", "message": "Odds snapshot has no captured_at timestamp."}

    ts = pd.to_datetime(odds["captured_at"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return {"status": "unknown", "message": "Could not parse captured_at timestamp."}

    latest = ts.max()
    age_hours = (datetime.now(UTC) - latest.to_pydatetime()).total_seconds() / 3600.0
    return {
        "status": "ok",
        "message": f"Latest odds snapshot: {latest.strftime('%Y-%m-%d %H:%M UTC')} ({age_hours:.1f}h ago)",
        "captured_at": latest.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "age_hours": age_hours,
        "rows": int(len(odds)),
    }


def _render_match_card(row: pd.Series) -> None:
    match_title = f"{row.get('p1_name', 'Player 1')} vs {row.get('p2_name', 'Player 2')}"
    tournament = row.get("tournament")
    surface = row.get("surface")
    round_name = row.get("round")
    meta_parts = [str(row.get("tour", "")).upper(), str(row.get("match_date", ""))]
    if pd.notna(tournament):
        meta_parts.append(str(tournament))
    if pd.notna(surface):
        meta_parts.append(str(surface))
    if pd.notna(round_name):
        meta_parts.append(str(round_name))

    pick_name = row.get("p1_name") if str(row.get("bet_side")) == "P1" else row.get("p2_name")

    with st.container(border=True):
        st.markdown(f"#### {match_title}")
        st.caption(" | ".join(part for part in meta_parts if part and part != ""))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Bet side", str(pick_name))
        col2.metric("Recommended stake", _format_currency(row.get("recommended_stake")))
        col3.metric("Model probability", _format_pct(row.get("selected_prob"), digits=1))
        col4.metric("Odds", _format_decimal(row.get("selected_odds"), digits=2))

        st.write(
            f"Edge: {_format_pct(row.get('selected_edge'), digits=1)} ({_edge_band(row.get('selected_edge'))}) | "
            f"Confidence: {_confidence_badge(row.get('confidence_tier'))} | "
            f"Agreement: {_format_pct(row.get('model_agreement'), digits=1)} | "
            f"EV per EUR: {_format_pct(row.get('expected_value_per_euro'), digits=1)}"
        )


def _render_recommendations(
    tour_filter: str,
    capital: float,
    min_edge_threshold: float,
    max_daily_bets: int,
) -> dict[str, dict[str, Any]]:
    total_analyzed = 0
    results: dict[str, dict[str, Any]] = {}
    combined_value_rows: list[pd.DataFrame] = []
    closest_rows: list[pd.DataFrame] = []

    for tour in _tour_list(tour_filter):
        pred = _load_prediction_file(tour)
        total_analyzed += len(pred)

        pred_path = PROCESSED_DIR / f"{tour}_predictions.csv"
        if pred.empty or not pred_path.exists():
            results[tour] = {
                "status": "missing_predictions",
                "message": f"{tour.upper()}: no prediction file yet. Click Refresh Predictions.",
                "recommendations": pd.DataFrame(),
                "analyzed_matches": 0,
            }
            continue

        rec = generate_recommendations(
            predictions_path=pred_path,
            odds_path=ODDS_UPCOMING_FILE,
            capital=capital,
            min_edge_threshold=min_edge_threshold,
            max_daily_bets=max_daily_bets,
            max_daily_capital_pct=MAX_DAILY_CAPITAL_PCT,
        )
        rec["analyzed_matches"] = len(pred)
        results[tour] = rec

        if rec.get("status") == "value":
            table = rec.get("recommendations", pd.DataFrame()).copy()
            if not table.empty:
                table["tour"] = tour
                combined_value_rows.append(table)
        elif rec.get("status") == "skip":
            closest = rec.get("closest", pd.DataFrame())
            if isinstance(closest, pd.DataFrame) and not closest.empty:
                closest = closest.copy()
                closest["tour"] = tour
                closest_rows.append(closest)

    total_value_bets = sum(
        len(rec.get("recommendations", pd.DataFrame()))
        for rec in results.values()
        if isinstance(rec.get("recommendations"), pd.DataFrame)
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Analyzed matches", total_analyzed)
    col2.metric("Value bets", total_value_bets)

    if combined_value_rows:
        combined = pd.concat(combined_value_rows, ignore_index=True).sort_values("selected_edge", ascending=False)
        total_stake = float(pd.to_numeric(combined["recommended_stake"], errors="coerce").fillna(0.0).sum())
        allocation_pct = (total_stake / capital) if capital > 0 else 0.0
        col3.metric("Recommended allocation", f"{_format_pct(allocation_pct, digits=1)} / {_format_currency(total_stake)}")

        st.success(
            f"BETTING SIGNAL: {len(combined)} value bet(s) detected. "
            f"Threshold {_format_pct(min_edge_threshold, digits=1)}."
        )
        for _, row in combined.iterrows():
            _render_match_card(row)
    else:
        col3.metric("Recommended allocation", "0.0% / EUR 0.00")
        if total_analyzed > 0:
            st.error(
                f"SKIP TODAY - No value bets detected across {total_analyzed} analyzed matches "
                f"at the {_format_pct(min_edge_threshold, digits=1)} threshold."
            )
            if closest_rows:
                closest = pd.concat(closest_rows, ignore_index=True).sort_values("selected_edge", ascending=False).iloc[0]
                pick_name = closest.get("p1_name") if str(closest.get("bet_side")) == "P1" else closest.get("p2_name")
                st.info(
                    f"Closest to value: {closest.get('p1_name')} vs {closest.get('p2_name')} "
                    f"({str(closest.get('tour', '')).upper()}) on {closest.get('match_date')} | "
                    f"lean {pick_name} at {_format_pct(closest.get('selected_edge'), digits=1)} edge."
                )
        else:
            st.info("No predicted matches available yet. Refresh odds and predictions first.")

    for tour in _tour_list(tour_filter):
        rec = results.get(tour, {})
        if rec.get("status") == "missing_predictions":
            st.caption(rec["message"])
        elif rec.get("status") == "no_matches":
            st.caption(f"{tour.upper()}: {rec.get('message')}")

    return results


def _log_recommendations_to_prediction_log(recommendation_results: dict[str, dict[str, Any]]) -> int:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_df = _load_prediction_log()
    existing_ids = set(log_df["prediction_id"].astype("string").fillna("").tolist())
    new_rows: list[dict[str, Any]] = []

    for tour, rec in recommendation_results.items():
        if rec.get("status") != "value":
            continue
        table = rec.get("recommendations")
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        for _, row in table.iterrows():
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


def _build_calibration_chart(calibration: pd.DataFrame, tour: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=calibration["avg_pred"],
            y=calibration["avg_true"],
            mode="lines+markers",
            name=f"{tour.upper()} observed",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"dash": "dash"},
        )
    )
    fig.update_layout(
        height=320,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis_title="Predicted probability",
        yaxis_title="Observed win rate",
    )
    return fig


def _build_feature_importance_chart(importance: pd.DataFrame, tour: str) -> go.Figure:
    if importance.empty:
        return go.Figure()

    pieces = []
    for model_name in sorted(importance["model"].astype("string").dropna().unique()):
        subset = importance[importance["model"].astype("string") == model_name].copy()
        subset["importance"] = pd.to_numeric(subset["importance"], errors="coerce")
        subset = subset.dropna(subset=["importance"]).sort_values("importance", ascending=False).head(8)
        pieces.append(subset)

    chart_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if chart_df.empty:
        return go.Figure()

    fig = px.bar(
        chart_df.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        color="model",
        orientation="h",
        barmode="group",
        title=f"{tour.upper()} feature importance",
    )
    fig.update_layout(height=420, margin={"l": 20, "r": 20, "t": 40, "b": 20}, yaxis_title="")
    return fig


def _load_equity_curve(tour: str, strategy: str) -> pd.DataFrame:
    return _safe_read_csv(MODELS_DIR / f"backtest_equity_{tour}_{strategy}.csv")


def _build_file_status() -> pd.DataFrame:
    items = [
        ("ATP master", PROCESSED_DIR / "atp_matches_master.csv"),
        ("WTA master", PROCESSED_DIR / "wta_matches_master.csv"),
        ("ATP predictions", PROCESSED_DIR / "atp_predictions.csv"),
        ("WTA predictions", PROCESSED_DIR / "wta_predictions.csv"),
        ("Upcoming odds", ODDS_UPCOMING_FILE),
        ("Odds history", ODDS_HISTORY_FILE),
        ("Prediction log", PREDICTION_LOG_FILE),
        ("Bankroll log", BANKROLL_LOG_FILE),
    ]
    rows = []
    for label, path in items:
        exists = path.exists()
        try:
            rows_count = len(pd.read_csv(path, low_memory=False)) if exists and path.suffix == ".csv" and path.stat().st_size > 0 else pd.NA
        except Exception:
            rows_count = pd.NA
        modified = (
            datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            if exists
            else "missing"
        )
        rows.append(
            {
                "artifact": label,
                "path": str(path),
                "exists": exists,
                "rows": rows_count,
                "last_modified": modified,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    _bootstrap_files()
    bankroll_state = _load_bankroll_state()
    stored_capital = float(bankroll_state.get("capital", DEFAULT_CAPITAL))

    st.set_page_config(page_title="TennisBet", layout="wide")
    st.title("TennisBet")
    st.caption(
        "Statistical pre-match betting assistant. Past performance does not guarantee future results. Gamble responsibly."
    )

    state = _load_last_update()
    odds_status = _odds_snapshot_status()

    with st.sidebar:
        st.header("Controls")
        capital = st.number_input("Current capital (EUR)", min_value=0.0, value=float(stored_capital), step=0.5)
        tour_filter = st.selectbox("Tour", ["both", "atp", "wta"], index=0)
        min_edge_threshold = st.number_input(
            "Min edge threshold",
            min_value=0.0,
            max_value=0.5,
            value=float(MIN_EDGE_THRESHOLD),
            step=0.01,
        )
        max_daily_bets = int(
            st.number_input(
                "Max daily bets",
                min_value=1,
                max_value=10,
                value=MAX_DAILY_BETS,
                step=1,
            )
        )

        if st.button("Update Data", type="primary"):
            try:
                with st.spinner("Pulling data, rebuilding pipeline, updating ELO..."):
                    update_report, pipeline_report, elo_report = _run_data_refresh()
                st.success("Data refresh completed.")
                state = _load_last_update()
                st.json({"update": update_report, "pipeline": pipeline_report, "elo": elo_report})
            except Exception as exc:
                st.error(f"Data refresh failed: {exc}")

        if st.button("Refresh Odds"):
            try:
                with st.spinner("Refreshing odds (Flashscore fallback + normalization)..."):
                    odds_report = refresh_odds()
                st.success("Odds refresh completed.")
                odds_status = _odds_snapshot_status()
                st.json(odds_report)
            except Exception as exc:
                st.error(f"Odds refresh failed: {exc}")

        if st.button("Refresh Predictions"):
            try:
                with st.spinner("Generating predictions from latest feature tables..."):
                    outputs = _refresh_predictions(tour_filter=tour_filter)
                if outputs:
                    st.success("Predictions refreshed.")
                    st.write("\n".join(outputs))
                else:
                    st.warning("No feature files found to predict.")
            except Exception as exc:
                st.error(f"Prediction refresh failed: {exc}")

        st.divider()
        with st.expander("Manual Odds Upload"):
            st.caption(
                "Required columns: match_date, player_1, player_2, odds_p1, odds_p2. "
                "If the CSV has no tour column, the current sidebar tour is used unless it is set to both."
            )
            uploaded_odds = st.file_uploader("Upload odds CSV", type=["csv"], key="manual_odds_csv")
            if st.button("Import odds CSV", key="import_odds_csv"):
                if uploaded_odds is None:
                    st.error("Choose an odds CSV first.")
                else:
                    default_tour = None if tour_filter == "both" else tour_filter
                    ok, msg = _ingest_uploaded_odds_csv(uploaded_odds, default_tour=default_tour)
                    if ok:
                        st.success(msg)
                        odds_status = _odds_snapshot_status()
                    else:
                        st.error(msg)

        st.divider()
        with st.expander("Add Recent Match"):
            entry_tour = st.selectbox("Entry tour", ["atp", "wta"], key="entry_tour")
            with st.form("custom_match_form"):
                match_date = st.date_input("Date", value=datetime.now(UTC).date())
                col1, col2 = st.columns(2)
                with col1:
                    tourney_name = st.text_input("Tournament", value="")
                    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet", "Unknown"])
                    round_name = st.text_input("Round", value="R32")
                    winner_name = st.text_input("Winner name", value="")
                    winner_id = st.text_input("Winner ID", value="")
                with col2:
                    tourney_level = st.text_input("Level", value="A")
                    best_of = st.number_input("Best of", min_value=3, max_value=5, value=3, step=2)
                    loser_name = st.text_input("Loser name", value="")
                    loser_id = st.text_input("Loser ID", value="")
                    score = st.text_input("Score", value="6-4 6-4")

                submitted = st.form_submit_button("Save custom match")
                if submitted:
                    payload = {
                        "tourney_id": f"custom-{entry_tour}-{match_date.strftime('%Y%m%d')}",
                        "tourney_name": tourney_name,
                        "surface": surface,
                        "draw_size": pd.NA,
                        "tourney_level": tourney_level,
                        "tourney_date": match_date.strftime("%Y%m%d"),
                        "match_num": int(datetime.now(UTC).timestamp()) % 1000000,
                        "winner_id": winner_id,
                        "winner_name": winner_name,
                        "loser_id": loser_id,
                        "loser_name": loser_name,
                        "score": score,
                        "best_of": best_of,
                        "round": round_name,
                        "source": "custom",
                    }

                    if not all([tourney_name, winner_name, loser_name, winner_id, loser_id]):
                        st.error("Tournament, player names, and player IDs are required.")
                    elif winner_id == loser_id:
                        st.error("Winner ID and loser ID must be different.")
                    else:
                        known_ids = _known_player_ids(entry_tour)
                        missing_ids = [pid for pid in (winner_id, loser_id) if pid not in known_ids]
                        if missing_ids:
                            st.error(
                                f"Unknown {entry_tour.upper()} player ID(s): {', '.join(missing_ids)}. "
                                "Use IDs from the current player reference file."
                            )
                        else:
                            _append_custom_match(entry_tour, payload)
                            st.success("Custom match saved.")

            uploaded_custom = st.file_uploader("Upload custom CSV", type=["csv"], key="custom_match_csv")
            if uploaded_custom is not None and st.button("Import custom CSV", key="import_custom_csv"):
                ok, msg = _ingest_uploaded_custom_csv(entry_tour, uploaded_custom)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.divider()
        with st.expander("Log Today's Results"):
            log_df = _load_prediction_log()
            open_df = log_df[log_df["status"].astype("string").fillna("").str.lower() == "open"].copy()
            if open_df.empty:
                st.info("No open predictions to settle.")
            else:
                open_df["label"] = open_df.apply(
                    lambda r: (
                        f"[{str(r.get('tour', '')).upper()}] {r.get('match_date')} | "
                        f"{r.get('match')} | {r.get('bet_side')} | stake EUR {r.get('stake')}"
                    ),
                    axis=1,
                )
                selected_label = st.selectbox("Open prediction", open_df["label"].tolist(), key="settle_prediction")
                selected_row = open_df[open_df["label"] == selected_label].iloc[0]
                default_odds = float(selected_row["odds"]) if pd.notna(selected_row["odds"]) else 2.0
                outcome = st.selectbox("Outcome", ["win", "loss", "void"], key="settle_outcome")
                settled_odds = st.number_input(
                    "Settled odds",
                    min_value=1.01,
                    value=default_odds,
                    step=0.01,
                    key="settled_odds",
                )
                if st.button("Apply Result"):
                    stake = float(selected_row["stake"]) if pd.notna(selected_row["stake"]) else 0.0
                    if outcome == "win":
                        pnl = round(stake * (float(settled_odds) - 1.0), 2)
                    elif outcome == "void":
                        pnl = 0.0
                    else:
                        pnl = round(-stake, 2)

                    mask = log_df["prediction_id"].astype("string") == str(selected_row["prediction_id"])
                    log_df.loc[mask, "result"] = outcome
                    log_df.loc[mask, "pnl"] = pnl
                    log_df.loc[mask, "odds"] = settled_odds
                    log_df.loc[mask, "status"] = "closed"
                    log_df.loc[mask, "resolved_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                    _save_prediction_log(log_df)

                    entry = _apply_bankroll_update(
                        pnl=pnl,
                        prediction_id=str(selected_row["prediction_id"]),
                        note=f"Settled {selected_row['match']} as {outcome}",
                    )
                    st.success(
                        f"Result saved. PnL EUR {pnl:.2f}. "
                        f"Capital: EUR {entry['capital_before']:.2f} -> EUR {entry['capital_after']:.2f}"
                    )

        st.divider()
        st.write(f"Last updated: {state.get('updated_at', 'never')}")
        st.write(f"Matches in DB: ATP {_count_matches('atp')} | WTA {_count_matches('wta')}")
        st.write(f"Model trained: {state.get('model_last_trained', 'not trained')}")
        st.write(f"Bankroll tracked: EUR {_load_bankroll_state().get('capital', DEFAULT_CAPITAL):.2f}")
        st.write(f"Odds status: {odds_status.get('message', 'unknown')}")

        latest_metrics = _load_model_metrics()
        st.caption("Model snapshot")
        if latest_metrics.empty:
            st.caption("No metrics available.")
        else:
            latest_metrics = latest_metrics.sort_values("trained_at").drop_duplicates(subset=["tour"], keep="last")
            for _, row in latest_metrics.iterrows():
                st.write(
                    f"{str(row.get('tour', '')).upper()}: acc {_format_decimal(row.get('ensemble_accuracy'), 3)} | "
                    f"log-loss {_format_decimal(row.get('ensemble_log_loss'), 3)}"
                )

    _show_staleness_banner(state)
    if odds_status.get("status") == "ok" and float(odds_status.get("age_hours", 0.0)) > 4.0:
        st.warning(
            f"Odds are {float(odds_status.get('age_hours', 0.0)):.1f} hours old. "
            "Refresh odds before placing bets if the market may have moved."
        )

    tab1, tab2, tab3 = st.tabs(["Today's Recommendations", "Model Performance", "Data Status"])

    with tab1:
        st.subheader("Today's Recommendations")
        st.caption(
            f"{datetime.now(UTC).strftime('%Y-%m-%d')} | "
            f"{_count_predicted_matches(tour_filter)} predicted match(es) currently available"
        )
        st.write(
            f"Tour filter: {tour_filter} | Capital: {_format_currency(capital)} | "
            f"Threshold: {_format_pct(min_edge_threshold, digits=1)} | Max bets: {max_daily_bets}"
        )
        rec_results = _render_recommendations(
            tour_filter=tour_filter,
            capital=float(capital),
            min_edge_threshold=float(min_edge_threshold),
            max_daily_bets=max_daily_bets,
        )
        if st.button("Log Current Recommendations"):
            count = _log_recommendations_to_prediction_log(rec_results)
            if count > 0:
                st.success(f"Logged {count} recommendation(s) to prediction log.")
            else:
                st.info("No new recommendations to log.")

    with tab2:
        st.subheader("Model Performance")
        metrics = _load_model_metrics()
        tours = _tour_list(tour_filter)

        if metrics.empty:
            st.info("No model metrics found yet. Run training first.")
        else:
            latest = metrics.sort_values("trained_at").drop_duplicates(subset=["tour"], keep="last")
            for tour in tours:
                section = latest[latest["tour"].astype("string").str.lower() == tour]
                if section.empty:
                    st.caption(f"{tour.upper()}: no model metrics available.")
                    continue
                row = section.iloc[0]
                st.markdown(f"### {tour.upper()}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", _format_pct(row.get("ensemble_accuracy"), digits=1))
                col2.metric("Log-loss", _format_decimal(row.get("ensemble_log_loss"), digits=3))
                col3.metric("ECE", _format_decimal(row.get("ensemble_ece"), digits=3))
                col4.metric("Test rows", f"{int(float(row.get('rows_test', 0))):,}")

                calibration = _load_calibration(tour)
                if not calibration.empty and {"avg_pred", "avg_true"}.issubset(calibration.columns):
                    st.plotly_chart(_build_calibration_chart(calibration, tour), use_container_width=True)
                else:
                    st.caption(f"{tour.upper()}: calibration data not available.")

                importance = _load_feature_importance(tour)
                if not importance.empty and {"model", "feature", "importance"}.issubset(importance.columns):
                    st.plotly_chart(_build_feature_importance_chart(importance, tour), use_container_width=True)
                else:
                    st.caption(f"{tour.upper()}: feature importance data not available.")

        backtest = _load_backtest_results()
        st.markdown("### Backtest")
        if backtest.empty:
            st.info("No backtest results yet.")
        else:
            filtered = backtest[backtest["tour"].astype("string").str.lower().isin(tours)].copy() if "tour" in backtest.columns else backtest.copy()
            st.dataframe(filtered, use_container_width=True)

            equity_frames = []
            for tour in tours:
                for strategy in ("value_dynamic", "flat_favorite", "flat_random"):
                    equity = _load_equity_curve(tour, strategy)
                    if equity.empty or "match_date" not in equity.columns or "equity" not in equity.columns:
                        continue
                    equity = equity.copy()
                    equity["tour"] = tour.upper()
                    equity["strategy"] = strategy
                    equity_frames.append(equity)

            if equity_frames:
                equity_df = pd.concat(equity_frames, ignore_index=True)
                fig = px.line(
                    equity_df.dropna(subset=["match_date", "equity"]),
                    x="match_date",
                    y="equity",
                    color="strategy",
                    line_dash="tour",
                    title="Backtest equity curve",
                )
                fig.update_layout(height=360, margin={"l": 20, "r": 20, "t": 40, "b": 20})
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Recent Prediction Log")
        pred_log = _load_prediction_log()
        if pred_log.empty:
            st.info("Prediction log is empty.")
        else:
            filtered_log = pred_log[pred_log["tour"].astype("string").str.lower().isin(tours)].copy()
            display_cols = [
                "created_at",
                "match_date",
                "tour",
                "match",
                "bet_side",
                "probability",
                "odds",
                "edge",
                "stake",
                "status",
                "result",
                "pnl",
            ]
            st.dataframe(filtered_log.tail(30)[display_cols], use_container_width=True)

    with tab3:
        st.subheader("Data Status")
        bankroll_state = _load_bankroll_state()
        pred_log = _load_prediction_log()
        open_bets = int(pred_log["status"].astype("string").fillna("").str.lower().eq("open").sum()) if not pred_log.empty else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last update", state.get("updated_at", "never"))
        col2.metric("Last new match", str(state.get("last_new_match", "unknown")))
        col3.metric("Odds rows", str(odds_status.get("rows", 0)))
        col4.metric("Open bets", str(open_bets))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("ATP custom rows", str(_count_custom_matches("atp")))
        col6.metric("WTA custom rows", str(_count_custom_matches("wta")))
        col7.metric("Bankroll", _format_currency(bankroll_state.get("capital", DEFAULT_CAPITAL)))
        col8.metric("Odds age", f"{float(odds_status.get('age_hours', 0.0)):.1f}h" if odds_status.get("status") == "ok" else "n/a")

        st.markdown("### Core Artifacts")
        st.dataframe(_build_file_status(), use_container_width=True)

        latest_predictions = []
        for tour in _tour_list(tour_filter):
            pred = _load_prediction_file(tour)
            if not pred.empty:
                latest_predictions.append(pred.assign(tour=tour).tail(10))
        if latest_predictions:
            st.markdown("### Latest Predictions")
            latest_df = pd.concat(latest_predictions, ignore_index=True)
            st.dataframe(
                latest_df[
                    [
                        "match_date",
                        "tour",
                        "p1_name",
                        "p2_name",
                        "ensemble_prob_p1",
                        "confidence_tier",
                        "model_agreement",
                    ]
                ],
                use_container_width=True,
            )

        with st.expander("Raw status payload"):
            st.json(state)


if __name__ == "__main__":
    main()
