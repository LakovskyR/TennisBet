from __future__ import annotations

import argparse
from functools import lru_cache
import json
import logging
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    MODELS_DIR,
    ODDS_UPCOMING_FILE,
    PROCESSED_DIR,
    RAW_ATP,
    RAW_WTA,
)
from src.player_aliases import canonicalize_player_name, load_player_aliases, normalize_player_name
from src.sqlite_storage import load_matches_frame, load_odds_frame

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import Booster as LGBMBooster
except Exception:  # pragma: no cover
    LGBMBooster = None


logger = logging.getLogger("predictor")
DEFAULT_ENSEMBLE_MODEL_ORDER = ("catboost", "xgboost", "lgbm")
DEFAULT_ENSEMBLE_CONFIG = {
    "winner": "top3_equal",
    "weights": {model_name: 1 / 3 for model_name in DEFAULT_ENSEMBLE_MODEL_ORDER},
}


def _safe_json_load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text) if text.strip() else {}


@lru_cache(maxsize=2)
def _load_uncertainty_config(tour: str) -> dict[str, Any]:
    path = MODELS_DIR / f"uncertainty_{tour}.json"
    if not path.exists():
        return {}
    try:
        return _safe_json_load(path)
    except Exception:
        return {}


def _model_artifact_path(tour: str, model_name: str) -> Path:
    suffixes = {
        "catboost": f"catboost_{tour}.cbm",
        "xgboost": f"xgboost_{tour}.json",
        "lgbm": f"lgbm_{tour}.txt",
        "rf": f"rf_{tour}.pkl",
        "elasticnet": f"elasticnet_{tour}.pkl",
        "logreg": f"logreg_{tour}.pkl",
        "ridge": f"ridge_{tour}.pkl",
    }
    return MODELS_DIR / suffixes[model_name]


def _equal_weight_config(model_names: tuple[str, ...] = DEFAULT_ENSEMBLE_MODEL_ORDER) -> dict[str, Any]:
    active_models = tuple(dict.fromkeys(str(name) for name in model_names if str(name)))
    if not active_models:
        active_models = DEFAULT_ENSEMBLE_MODEL_ORDER
    weight = 1.0 / len(active_models)
    return {
        "winner": "_".join(active_models) + "_equal",
        "weights": {model_name: weight for model_name in active_models},
    }


def _extract_ensemble_config(payload: dict[str, Any]) -> dict[str, Any]:
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        return {}
    return {
        "winner": str(payload.get("winner") or payload.get("name") or "ensemble"),
        "weights": {str(name): float(weight) for name, weight in weights.items()},
        **{
            key: payload[key]
            for key in ("cv_log_loss", "cv_brier", "cv_accuracy", "selection_date")
            if key in payload
        },
    }


def _load_report_ensemble_config(tour: str) -> dict[str, Any]:
    path = MODELS_DIR / f"model_report_{tour}.json"
    if not path.exists():
        return {}
    try:
        payload = _safe_json_load(path)
    except Exception:
        return {}
    for candidate in (
        payload.get("ensemble_config"),
        payload.get("temporal_cv", {}).get("winner_row"),
        payload.get("winner_row"),
    ):
        if isinstance(candidate, dict):
            config = _extract_ensemble_config(candidate)
            if config:
                return config
    return {}


def _load_ensemble_config(tour: str) -> dict[str, Any]:
    path = MODELS_DIR / f"ensemble_config_{tour}.json"
    if path.exists():
        try:
            payload = _safe_json_load(path)
        except Exception:
            payload = {}
        config = _extract_ensemble_config(payload) if isinstance(payload, dict) else {}
        if config:
            return config
    report_config = _load_report_ensemble_config(tour)
    if report_config:
        return report_config
    return _equal_weight_config()


def _sanitize_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in cols:
        base = re.sub(r"[\[\]<>]", "_", str(col))
        base = re.sub(r"\s+", "_", base)
        base = re.sub(r"_+", "_", base).strip("_")
        if not base:
            base = "feature"
        count = seen.get(base, 0)
        name = base if count == 0 else f"{base}_{count}"
        seen[base] = count + 1
        out.append(name)
    return out


def _is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()
            second_line = handle.readline().strip()
    except OSError:
        return False
    return first_line == "version https://git-lfs.github.com/spec/v1" and second_line.startswith("oid sha256:")


@lru_cache(maxsize=8)
def _lightgbm_artifact_is_loadable(path_str: str) -> bool:
    path = Path(path_str)
    if not path.exists() or path.stat().st_size == 0:
        return False
    probe = (
        f"from lightgbm import Booster\n"
        f"Booster(model_file=r'''{path}''')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _load_single_model(model_name: str, path: Path) -> Any | None:
    if not path.exists():
        logger.warning("Missing %s artifact: %s", model_name, path)
        return None
    if _is_git_lfs_pointer(path):
        logger.warning("Skipping %s artifact stored as Git LFS pointer: %s", model_name, path)
        return None
    try:
        if model_name == "catboost":
            if CatBoostClassifier is None:
                logger.warning("CatBoost not installed; skipping catboost")
                return None
            model = CatBoostClassifier()
            model.load_model(str(path))
            return model
        if model_name == "xgboost":
            if XGBClassifier is None:
                logger.warning("XGBoost not installed; skipping xgboost")
                return None
            model = XGBClassifier()
            model.load_model(str(path))
            return model
        if model_name == "lgbm":
            if LGBMBooster is None:
                logger.warning("LightGBM not installed; skipping lgbm")
                return None
            if not _lightgbm_artifact_is_loadable(str(path.resolve())):
                logger.warning("Skipping invalid LightGBM artifact: %s", path)
                return None
            with tempfile.NamedTemporaryFile(prefix="tennisbet_lgbm_", suffix=".txt", delete=False) as handle:
                temp_path = Path(handle.name)
            try:
                shutil.copyfile(path, temp_path)
                return LGBMBooster(model_file=str(temp_path))
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        logger.warning("Skipping unloadable %s artifact %s: %s", model_name, path.name, exc)
        return None


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(float(v) for v in weights.values()))
    if total <= 0:
        return {}
    return {name: float(value) / total for name, value in weights.items()}


def _load_models_and_schema(tour: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prep_path = MODELS_DIR / f"preprocess_{tour}.json"
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing preprocess artifact for {tour}: {prep_path.name}.")

    ensemble_config = _load_ensemble_config(tour)
    weights = dict(ensemble_config.get("weights", {}))
    models: dict[str, Any] = {}
    for model_name in list(weights.keys()):
        model = _load_single_model(model_name, _model_artifact_path(tour, model_name))
        if model is None:
            continue
        models[model_name] = model

    if not models and ensemble_config.get("winner") != DEFAULT_ENSEMBLE_CONFIG["winner"]:
        ensemble_config = _default_ensemble_config(tour)
        for model_name in list(ensemble_config["weights"].keys()):
            model = _load_single_model(model_name, _model_artifact_path(tour, model_name))
            if model is not None:
                models[model_name] = model

    if not models:
        raise FileNotFoundError(f"No loadable model artifacts found for {tour}.")

    schema = _safe_json_load(prep_path)
    resolved_weights = _normalize_weights({name: float(weights.get(name, ensemble_config["weights"].get(name, 0.0))) for name in models})
    if not resolved_weights:
        resolved_weights = _normalize_weights({name: float(weight) for name, weight in ensemble_config["weights"].items() if name in models})
    ensemble_config["weights"] = resolved_weights
    return models, ensemble_config, schema


def _prepare_for_models(df: pd.DataFrame, schema: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols: list[str] = list(schema.get("feature_cols", []))
    cat_cols: list[str] = list(schema.get("cat_cols", []))
    num_cols: list[str] = list(schema.get("num_cols", []))
    medians: dict[str, float] = dict(schema.get("numeric_medians", {}))
    xgb_feature_cols: list[str] = list(schema.get("xgb_feature_cols", []))

    if not feature_cols:
        raise ValueError("Invalid preprocess schema: feature_cols missing")

    x = df.copy()
    for col in feature_cols:
        if col not in x.columns:
            x[col] = pd.NA

    x = x[feature_cols].copy()

    for col in cat_cols:
        x[col] = x[col].astype("string").fillna("Unknown")

    for col in num_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        x[col] = x[col].fillna(float(medians.get(col, 0.0)))

    xgb = pd.get_dummies(x, columns=cat_cols, dummy_na=True)
    xgb_cols_original = list(xgb.columns)
    xgb.columns = _sanitize_columns(xgb_cols_original)

    if xgb_feature_cols:
        xgb = xgb.reindex(columns=xgb_feature_cols, fill_value=0)

    return x, xgb


def _predict_model_proba(model_name: str, model: Any, x_cat: pd.DataFrame, x_tab: pd.DataFrame) -> np.ndarray:
    if model_name == "catboost":
        return model.predict_proba(x_cat)[:, 1]
    if model_name == "lgbm":
        preds = model.predict(x_tab)
        return np.asarray(preds, dtype=float)
    return model.predict_proba(x_tab)[:, 1]


def add_prediction_columns(df: pd.DataFrame, tour: str) -> pd.DataFrame:
    models, ensemble_config, schema = _load_models_and_schema(tour)
    x_cat, x_xgb = _prepare_for_models(df, schema)

    probability_map = {
        model_name: _predict_model_proba(model_name, model, x_cat, x_xgb)
        for model_name, model in models.items()
    }
    stacked = np.vstack([np.asarray(values, dtype=float) for values in probability_map.values()])
    if stacked.shape[0] == 1:
        agreement_gap = np.zeros(stacked.shape[1], dtype=float)
    else:
        agreement_gap = stacked.max(axis=0) - stacked.min(axis=0)
    confidence = np.where(
        agreement_gap < 0.05,
        "HIGH",
        np.where(agreement_gap < 0.12, "MEDIUM", "LOW"),
    )
    ensemble_prob = np.zeros(stacked.shape[1], dtype=float)
    for model_name, weight in ensemble_config.get("weights", {}).items():
        probs = probability_map.get(model_name)
        if probs is not None:
            ensemble_prob += float(weight) * probs
    uncertainty = _load_uncertainty_config(tour)
    interval_radius = float(uncertainty.get("residual_quantile", 0.0) or 0.0)
    interval_lower = np.clip(ensemble_prob - interval_radius, 0.0, 1.0)
    interval_upper = np.clip(ensemble_prob + interval_radius, 0.0, 1.0)

    out = df.copy()
    for model_name in ["catboost", "xgboost", "lgbm", "rf", "elasticnet", "logreg", "ridge"]:
        out[f"{model_name}_prob"] = probability_map.get(model_name, np.nan)
    out["ensemble_prob_p1"] = ensemble_prob
    out["ensemble_prob_p1_lower"] = interval_lower
    out["ensemble_prob_p1_upper"] = interval_upper
    out["ensemble_prob_p1_width"] = interval_upper - interval_lower
    out["model_agreement"] = 1.0 - agreement_gap
    out["confidence_tier"] = confidence

    return out


def _norm_name(name: Any) -> str:
    return normalize_player_name(name)


def _load_player_maps(tour: str) -> tuple[dict[str, str], dict[str, str]]:
    players_file = RAW_ATP / "atp_players.csv" if tour == "atp" else RAW_WTA / "wta_players.csv"
    if not players_file.exists():
        return {}, {}
    df = pd.read_csv(players_file, usecols=["player_id", "name_first", "name_last", "ioc"], low_memory=False)
    name_to_id: dict[str, str] = {}
    id_to_ioc: dict[str, str] = {}
    for pid, first, last, ioc in df.itertuples(index=False):
        full = f"{first} {last}".strip()
        if not full:
            continue
        try:
            player_id = str(int(float(pid)))
        except Exception:
            player_id = str(pid)
        name_to_id[_norm_name(full)] = player_id
        if isinstance(ioc, str) and ioc.strip():
            id_to_ioc[player_id] = ioc.strip().upper()
    return name_to_id, id_to_ioc


def _load_tournament_country_map() -> tuple[dict[str, str], dict[str, str]]:
    path = Path("tournament_country.json")
    if not path.exists():
        return {}, {}
    text = path.read_text(encoding="utf-8-sig")
    raw = json.loads(text) if text.strip() else {}
    exact = {str(k): str(v).upper() for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}
    norm = {_norm_name(k): v for k, v in exact.items()}
    return exact, norm


def _resolve_tournament_country(name: Any, exact: dict[str, str], norm: dict[str, str]) -> str | None:
    if not isinstance(name, str):
        return None
    if name in exact:
        return exact[name]
    return norm.get(_norm_name(name))


def _build_player_states(feature_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    required = {"match_date", "match_key", "p1_id", "p2_id"}
    if feature_df.empty or not required.issubset(feature_df.columns):
        return {}
    feature_df = feature_df.copy()
    feature_df["match_date"] = pd.to_datetime(feature_df["match_date"], errors="coerce")
    feature_df = feature_df[feature_df["match_date"].notna()].sort_values(["match_date", "match_key"])

    state: dict[str, dict[str, Any]] = {}

    def capture(row: pd.Series, prefix: str) -> dict[str, Any]:
        return {
            "elo_overall": row.get(f"{prefix}_elo_overall"),
            "elo_surface": row.get(f"{prefix}_elo_surface"),
            "win_pct_5": row.get(f"{prefix}_win_pct_5"),
            "win_pct_10": row.get(f"{prefix}_win_pct_10"),
            "win_pct_20": row.get(f"{prefix}_win_pct_20"),
            "win_pct_surface_10": row.get(f"{prefix}_win_pct_surface_10"),
            "current_win_streak": row.get(f"{prefix}_current_win_streak"),
            "current_lose_streak": row.get(f"{prefix}_current_lose_streak"),
            "streak_5": row.get(f"{prefix}_streak_5"),
            "title_count_12m": row.get(f"{prefix}_title_count_12m"),
            "home_win_pct": row.get(f"{prefix}_home_win_pct"),
            "matches_last_14d": row.get(f"{prefix}_matches_last_14d"),
            "sets_played_last_7d": row.get(f"{prefix}_sets_played_last_7d"),
            "ace_pct": row.get(f"{prefix}_ace_pct"),
            "first_serve_pct": row.get(f"{prefix}_1st_serve_pct"),
            "bp_save_pct": row.get(f"{prefix}_bp_save_pct"),
            "matches_played_before": row.get(f"{prefix}_matches_played_before"),
            "rank": row.get(f"{prefix}_rank"),
            "last_match_date": row.get("match_date"),
        }

    for _, row in feature_df.iterrows():
        p1 = str(row.get("p1_id"))
        p2 = str(row.get("p2_id"))
        if p1 and p1 != "nan":
            state[p1] = capture(row, "p1")
        if p2 and p2 != "nan":
            state[p2] = capture(row, "p2")
    return state


@lru_cache(maxsize=2)
def _load_h2h_match_history(tour: str) -> pd.DataFrame:
    df = load_matches_frame(
        tour,
        columns=["match_date", "winner_id", "loser_id", "surface"],
        fallback_to_csv=False,
    )
    if df.empty:
        return df
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df[df["match_date"].notna()].copy()


def _h2h_stats(tour: str, p1_id: str, p2_id: str, surface: str, before_date: pd.Timestamp) -> dict[str, float]:
    m = _load_h2h_match_history(tour)
    if m.empty:
        return {
            "h2h_p1_wins": 0,
            "h2h_p2_wins": 0,
            "h2h_total": 0,
            "h2h_p1_win_pct": 0.5,
            "h2h_surface_p1_wins": 0,
            "h2h_surface_p2_wins": 0,
        }
    m = m[m["match_date"] < before_date]
    w = m["winner_id"].astype("string")
    l = m["loser_id"].astype("string")
    mask = ((w == p1_id) & (l == p2_id)) | ((w == p2_id) & (l == p1_id))
    h = m[mask]
    if h.empty:
        return {
            "h2h_p1_wins": 0,
            "h2h_p2_wins": 0,
            "h2h_total": 0,
            "h2h_p1_win_pct": 0.5,
            "h2h_surface_p1_wins": 0,
            "h2h_surface_p2_wins": 0,
        }

    h_w = h["winner_id"].astype("string")
    p1_wins = int((h_w == p1_id).sum())
    p2_wins = int((h_w == p2_id).sum())
    hs = h[h["surface"].astype("string") == str(surface)]
    hs_w = hs["winner_id"].astype("string")
    return {
        "h2h_p1_wins": p1_wins,
        "h2h_p2_wins": p2_wins,
        "h2h_total": int(len(h)),
        "h2h_p1_win_pct": float(p1_wins / len(h)) if len(h) else 0.5,
        "h2h_surface_p1_wins": int((hs_w == p1_id).sum()),
        "h2h_surface_p2_wins": int((hs_w == p2_id).sum()),
    }


def predict_from_odds(
    tour: str,
    odds_path: Path = ODDS_UPCOMING_FILE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    if odds_path == ODDS_UPCOMING_FILE:
        odds = load_odds_frame(current_only=True, fallback_to_csv=False)
    else:
        if not odds_path.exists() or odds_path.stat().st_size == 0:
            return pd.DataFrame()
        odds = pd.read_csv(odds_path, low_memory=False)
    odds = odds[odds.get("tour", pd.Series(dtype=str)).astype("string").str.lower() == tour].copy()
    if odds.empty:
        return pd.DataFrame()

    feature_path = PROCESSED_DIR / f"{tour}_player_features.csv"
    if not feature_path.exists():
        return pd.DataFrame()
    hist = pd.read_csv(feature_path, low_memory=False)
    required_feature_cols = {"match_date", "match_key", "p1_id", "p2_id"}
    if not required_feature_cols.issubset(hist.columns):
        logger.warning("Skipping odds prediction for %s: invalid feature artifact %s", tour, feature_path)
        return pd.DataFrame()
    states = _build_player_states(hist)

    _, _, schema = _load_models_and_schema(tour)
    feature_cols: list[str] = list(schema.get("feature_cols", []))

    name_to_id, id_to_ioc = _load_player_maps(tour)
    aliases = load_player_aliases()
    t_exact, t_norm = _load_tournament_country_map()

    rows: list[dict[str, Any]] = []
    unresolved_names: list[str] = []
    for row in odds.to_dict(orient="records"):
        p1_name = row.get("player_1_resolved") or row.get("player_1")
        p2_name = row.get("player_2_resolved") or row.get("player_2")
        p1_id = row.get("player_1_id")
        p2_id = row.get("player_2_id")

        p1_lookup = canonicalize_player_name(p1_name, aliases)
        p2_lookup = canonicalize_player_name(p2_name, aliases)
        p1_id = str(int(float(p1_id))) if pd.notna(p1_id) and str(p1_id).strip() else name_to_id.get(p1_lookup)
        p2_id = str(int(float(p2_id))) if pd.notna(p2_id) and str(p2_id).strip() else name_to_id.get(p2_lookup)
        if not p1_id or not p2_id:
            unresolved_names.append(f"{p1_name} vs {p2_name}")
            continue

        s1 = states.get(p1_id, {})
        s2 = states.get(p2_id, {})
        match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
        if pd.isna(match_date):
            match_date = pd.Timestamp.utcnow().tz_localize(None)

        surface = str(row.get("surface") or "Hard")
        tournament = row.get("tournament")
        tournament_country = _resolve_tournament_country(tournament, t_exact, t_norm)

        p1_ioc = id_to_ioc.get(p1_id)
        p2_ioc = id_to_ioc.get(p2_id)
        p1_is_home = int(bool(p1_ioc and tournament_country and p1_ioc == tournament_country))
        p2_is_home = int(bool(p2_ioc and tournament_country and p2_ioc == tournament_country))
        if p1_is_home == 1 and p2_is_home == 0:
            home_flag = 1
        elif p2_is_home == 1 and p1_is_home == 0:
            home_flag = -1
        else:
            home_flag = 0

        def val(st: dict[str, Any], key: str, default: float) -> float:
            v = st.get(key, default)
            try:
                if pd.isna(v):
                    return default
            except Exception:
                pass
            try:
                return float(v)
            except Exception:
                return default

        p1_last = pd.to_datetime(s1.get("last_match_date"), errors="coerce")
        p2_last = pd.to_datetime(s2.get("last_match_date"), errors="coerce")
        p1_days = int((match_date - p1_last).days) if pd.notna(p1_last) else 14
        p2_days = int((match_date - p2_last).days) if pd.notna(p2_last) else 14

        h2h = _h2h_stats(tour, p1_id, p2_id, surface, match_date)

        feat: dict[str, Any] = {c: pd.NA for c in feature_cols}
        feat.update(
            {
                "match_key": f"upcoming-{tour}-{row.get('match_id') or _norm_name(str(p1_name)+str(p2_name))}",
                "match_date": str(row.get("match_date")),
                "tour": tour,
                "p1_id": p1_id,
                "p2_id": p2_id,
                "p1_name": p1_name,
                "p2_name": p2_name,
                "p1_rank": val(s1, "rank", 999),
                "p2_rank": val(s2, "rank", 999),
                "p1_elo_overall": val(s1, "elo_overall", 1500),
                "p2_elo_overall": val(s2, "elo_overall", 1500),
                "p1_elo_surface": val(s1, "elo_surface", val(s1, "elo_overall", 1500)),
                "p2_elo_surface": val(s2, "elo_surface", val(s2, "elo_overall", 1500)),
                "p1_win_pct_5": val(s1, "win_pct_5", 0.5),
                "p1_win_pct_10": val(s1, "win_pct_10", 0.5),
                "p1_win_pct_20": val(s1, "win_pct_20", 0.5),
                "p2_win_pct_5": val(s2, "win_pct_5", 0.5),
                "p2_win_pct_10": val(s2, "win_pct_10", 0.5),
                "p2_win_pct_20": val(s2, "win_pct_20", 0.5),
                "p1_win_pct_surface_10": val(s1, "win_pct_surface_10", 0.5),
                "p2_win_pct_surface_10": val(s2, "win_pct_surface_10", 0.5),
                "p1_current_win_streak": val(s1, "current_win_streak", 0),
                "p2_current_win_streak": val(s2, "current_win_streak", 0),
                "p1_current_lose_streak": val(s1, "current_lose_streak", 0),
                "p2_current_lose_streak": val(s2, "current_lose_streak", 0),
                "p1_streak_5": val(s1, "streak_5", 0),
                "p2_streak_5": val(s2, "streak_5", 0),
                "p1_tournament_wins_current": 0,
                "p2_tournament_wins_current": 0,
                "p1_title_count_12m": val(s1, "title_count_12m", 0),
                "p2_title_count_12m": val(s2, "title_count_12m", 0),
                "p1_is_home": p1_is_home,
                "p2_is_home": p2_is_home,
                "home_advantage_flag": home_flag,
                "p1_home_win_pct": val(s1, "home_win_pct", 0.5),
                "p2_home_win_pct": val(s2, "home_win_pct", 0.5),
                "p1_days_since_last_match": max(0, p1_days),
                "p2_days_since_last_match": max(0, p2_days),
                "p1_matches_last_14d": val(s1, "matches_last_14d", 0),
                "p2_matches_last_14d": val(s2, "matches_last_14d", 0),
                "p1_sets_played_last_7d": val(s1, "sets_played_last_7d", 0),
                "p2_sets_played_last_7d": val(s2, "sets_played_last_7d", 0),
                "p1_ace_pct": val(s1, "ace_pct", 0.06),
                "p2_ace_pct": val(s2, "ace_pct", 0.06),
                "p1_1st_serve_pct": val(s1, "first_serve_pct", 0.62),
                "p2_1st_serve_pct": val(s2, "first_serve_pct", 0.62),
                "p1_bp_save_pct": val(s1, "bp_save_pct", 0.60),
                "p2_bp_save_pct": val(s2, "bp_save_pct", 0.60),
                "surface": surface,
                "tournament_level": "Tour",
                "round": "R32",
                "best_of": "3",
                "p1_wins": 0,
                "p1_matches_played_before": val(s1, "matches_played_before", 0),
                "p2_matches_played_before": val(s2, "matches_played_before", 0),
                **h2h,
            }
        )
        feat["elo_diff_overall"] = float(feat["p1_elo_overall"]) - float(feat["p2_elo_overall"])
        feat["elo_diff_surface"] = float(feat["p1_elo_surface"]) - float(feat["p2_elo_surface"])
        rows.append(feat)

    if not rows:
        if unresolved_names:
            sample = ", ".join(unresolved_names[:5])
            logger.warning(
                "No %s predictions built from odds; unresolved player IDs for %d match(es): %s",
                tour,
                len(unresolved_names),
                sample,
            )
        return pd.DataFrame()

    if unresolved_names:
        sample = ", ".join(unresolved_names[:5])
        logger.warning("Skipped %d %s odds row(s) with unresolved player IDs: %s", len(unresolved_names), tour, sample)

    upcoming_features = pd.DataFrame(rows)
    pred = add_prediction_columns(upcoming_features, tour=tour)
    keep = [
        "match_date",
        "match_key",
        "tour",
        "p1_name",
        "p2_name",
        "ensemble_prob_p1",
        "ensemble_prob_p1_lower",
        "ensemble_prob_p1_upper",
        "ensemble_prob_p1_width",
        "catboost_prob",
        "xgboost_prob",
        "lgbm_prob",
        "rf_prob",
        "elasticnet_prob",
        "logreg_prob",
        "confidence_tier",
        "model_agreement",
    ]
    result = pred[keep].copy()
    if output_path is None:
        output_path = PROCESSED_DIR / f"{tour}_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def predict_from_feature_file(tour: str, input_path: Path, output_path: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    if limit and limit > 0:
        df = df.tail(limit).copy()

    pred = add_prediction_columns(df, tour=tour)

    keep = [
        "match_date",
        "match_key",
        "tour",
        "p1_name",
        "p2_name",
        "ensemble_prob_p1",
        "ensemble_prob_p1_lower",
        "ensemble_prob_p1_upper",
        "ensemble_prob_p1_width",
        "catboost_prob",
        "xgboost_prob",
        "lgbm_prob",
        "rf_prob",
        "elasticnet_prob",
        "logreg_prob",
        "confidence_tier",
        "model_agreement",
    ]
    for col in keep:
        if col not in pred.columns:
            pred[col] = pd.NA

    result = pred[keep]

    if output_path is None:
        output_path = PROCESSED_DIR / f"{tour}_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TennisBet ensemble predictions")
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    parser.add_argument("--input", default=None, help="Feature CSV path; defaults to data/processed/{tour}_player_features.csv")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=200, help="Use only latest N rows from input")
    parser.add_argument("--from-odds", action="store_true", help="Build predictions for upcoming matches from odds CSV")
    parser.add_argument("--odds-file", default=str(ODDS_UPCOMING_FILE), help="Odds CSV path for --from-odds mode")

    args = parser.parse_args()
    output_path = Path(args.output) if args.output else PROCESSED_DIR / f"{args.tour}_predictions.csv"
    if args.from_odds:
        pred = predict_from_odds(
            tour=args.tour,
            odds_path=Path(args.odds_file),
            output_path=output_path,
        )
    else:
        input_path = Path(args.input) if args.input else PROCESSED_DIR / f"{args.tour}_player_features.csv"
        pred = predict_from_feature_file(
            tour=args.tour,
            input_path=input_path,
            output_path=output_path,
            limit=args.limit,
        )
    print(
        json.dumps(
            {
                "tour": args.tour,
                "rows": int(len(pred)),
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
