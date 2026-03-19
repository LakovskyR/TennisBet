from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import LAST_UPDATE_FILE, MODELS_DIR, OPTUNA_TRIALS, PROCESSED_DIR, TRAIN_CUTOFF

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
    from lightgbm import LGBMClassifier
    from lightgbm import early_stopping as lgbm_early_stopping
except Exception:  # pragma: no cover
    LGBMBooster = None
    LGBMClassifier = None
    lgbm_early_stopping = None

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CAT_COLS = ["surface", "tournament_level", "round", "best_of"]
TARGET_COL = "p1_wins"
META_COLS = ["match_key", "match_date", "tour", "p1_id", "p2_id", "p1_name", "p2_name"]
DEFAULT_TEMPORAL_CV_FOLDS = 3
TRAINING_STATE_FILE = MODELS_DIR / "training_state.json"
RETRAIN_WEEKLY_DAYS = 7
RETRAIN_MATCH_THRESHOLD = 50
TUNING_FOLDS = 3
PREDICTION_INTERVAL_ALPHA = 0.10
DEFAULT_ENSEMBLE_WEIGHTS = {
    "winner": "cat60_xgb40",
    "weights": {"catboost": 3 / 5, "xgboost": 2 / 5},
}
BASE_MODEL_ORDER = ["catboost", "xgboost", "lgbm", "rf", "elasticnet", "logreg"]
BASE_MODEL_LABELS = {
    "catboost": "catboost",
    "xgboost": "xgboost",
    "lgbm": "lgbm",
    "rf": "rf",
    "elasticnet": "elasticnet",
    "logreg": "logreg",
}
log = logging.getLogger("model_training")


def _safe_json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    return json.loads(text)


def _safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _higher_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:  # pragma: no cover
        return float(np.quantile(values, q, interpolation="higher"))


def _build_prediction_interval_payload(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    *,
    alpha: float = PREDICTION_INTERVAL_ALPHA,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=float)
    clipped = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    residuals = np.abs(truth - clipped)
    n = int(residuals.size)
    conformal_q = min(1.0, math.ceil((n + 1) * (1 - alpha)) / max(n, 1))
    radius = _higher_quantile(residuals, conformal_q)
    lower = np.clip(clipped - radius, 0.0, 1.0)
    upper = np.clip(clipped + radius, 0.0, 1.0)
    coverage = float(np.mean((truth >= lower) & (truth <= upper))) if n else 0.0
    return {
        "method": "split_conformal_absolute_residual",
        "alpha": float(alpha),
        "confidence_level": float(1.0 - alpha),
        "residual_quantile": float(radius),
        "empirical_coverage": coverage,
        "mean_interval_width": float(np.mean(upper - lower)) if n else 0.0,
        "n_test": n,
    }


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _load_features(tour: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{tour}_player_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file for {tour}: {path}")

    df = pd.read_csv(path, low_memory=False)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Feature file missing target column '{TARGET_COL}': {path}")

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df[df["match_date"].notna()].copy()
    df = df.sort_values(["match_date", "match_key"], na_position="last")
    return df


def _feature_schema_signature(columns: list[str]) -> str:
    payload = json.dumps(list(columns), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_feature_metadata(tour: str) -> dict[str, Any]:
    path = PROCESSED_DIR / f"{tour}_player_features.csv"
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
        }

    header = pd.read_csv(path, nrows=0)
    columns = header.columns.tolist()
    date_df = pd.read_csv(path, usecols=["match_date"], low_memory=False)
    match_dates = pd.to_datetime(date_df["match_date"], errors="coerce")
    latest_date = match_dates.max()

    return {
        "exists": True,
        "path": str(path),
        "feature_rows": int(len(date_df)),
        "feature_columns": columns,
        "feature_schema_signature": _feature_schema_signature(columns),
        "feature_date_max": latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else None,
    }


def _artifacts_for_tour(tour: str) -> dict[str, Path]:
    return {
        "catboost_model": MODELS_DIR / f"catboost_{tour}.cbm",
        "xgboost_model": MODELS_DIR / f"xgboost_{tour}.json",
        "lgbm_model": MODELS_DIR / f"lgbm_{tour}.txt",
        "rf_model": MODELS_DIR / f"rf_{tour}.pkl",
        "elasticnet_model": MODELS_DIR / f"elasticnet_{tour}.pkl",
        "logreg_model": MODELS_DIR / f"logreg_{tour}.pkl",
        "ensemble_config": MODELS_DIR / f"ensemble_config_{tour}.json",
        "preprocess": MODELS_DIR / f"preprocess_{tour}.json",
        "report": MODELS_DIR / f"model_report_{tour}.json",
    }


def _best_params_path(tour: str, model: str) -> Path:
    return MODELS_DIR / f"best_params_{tour}_{model}.json"


def _load_best_params(tour: str, model: str) -> dict[str, Any]:
    payload = _safe_json_load(_best_params_path(tour, model))
    params = payload.get("params")
    return params if isinstance(params, dict) else {}


def _save_best_params(tour: str, model: str, score: float, params: dict[str, Any]) -> Path:
    path = _best_params_path(tour, model)
    _safe_json_dump(
        path,
        {
            "tour": tour,
            "model": model,
            "score": float(score),
            "params": params,
            "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    return path


def _artifacts_missing(tour: str) -> list[str]:
    artifacts = _artifacts_for_tour(tour)
    required = {"preprocess", "report", "ensemble_config"}
    ensemble_config = _safe_json_load(artifacts["ensemble_config"])
    active_models = list(ensemble_config.get("weights", {}).keys()) if ensemble_config else list(DEFAULT_ENSEMBLE_WEIGHTS["weights"].keys())
    for model_name in active_models:
        model_key = f"{model_name}_model"
        if model_key in artifacts:
            required.add(model_key)
    return [name for name in required if not artifacts[name].exists()]


def _load_training_state(path: Path = TRAINING_STATE_FILE) -> dict[str, Any]:
    return _safe_json_load(path)


def _build_bootstrap_training_state(tours: tuple[str, ...]) -> dict[str, Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    state: dict[str, Any] = {
        "version": 1,
        "bootstrapped_at": now,
        "updated_at": now,
        "tours": {},
    }

    for tour in tours:
        metadata = _collect_feature_metadata(tour)
        report = _safe_json_load(MODELS_DIR / f"model_report_{tour}.json")
        state["tours"][tour] = {
            "tour": tour,
            "trained_at": report.get("trained_at"),
            "feature_rows": metadata.get("feature_rows", 0),
            "feature_date_max": metadata.get("feature_date_max"),
            "feature_schema_signature": metadata.get("feature_schema_signature"),
            "report_file": str(MODELS_DIR / f"model_report_{tour}.json"),
            "bootstrapped": True,
        }

    return state


def _save_training_state_after_training(
    results: dict[str, Any],
    *,
    train_cutoff: str,
    fallback_latest_months: int,
    cv_folds: int,
    path: Path = TRAINING_STATE_FILE,
) -> dict[str, Any]:
    state = _load_training_state(path)
    if not state:
        state = {"version": 1, "tours": {}}

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    state["version"] = 1
    state["updated_at"] = now
    state["train_cutoff"] = train_cutoff
    state["fallback_latest_months"] = fallback_latest_months
    state["cv_folds"] = cv_folds

    for tour, report in results.items():
        metadata = _collect_feature_metadata(tour)
        state.setdefault("tours", {})
        state["tours"][tour] = {
            "tour": tour,
            "trained_at": report.get("trained_at", now),
            "feature_rows": metadata.get("feature_rows", report.get("rows_total", 0)),
            "feature_date_max": metadata.get("feature_date_max"),
            "feature_schema_signature": metadata.get("feature_schema_signature"),
            "report_file": str(MODELS_DIR / f"model_report_{tour}.json"),
            "metrics": report.get("metrics", {}).get("ensemble", {}),
            "train_cutoff": train_cutoff,
            "fallback_latest_months": fallback_latest_months,
            "cv_folds": cv_folds,
            "bootstrapped": False,
        }

    _safe_json_dump(path, state)
    return state


def evaluate_retrain_policy(
    tours: tuple[str, ...] = ("atp", "wta"),
    *,
    path: Path = TRAINING_STATE_FILE,
    write_bootstrap_state: bool = False,
    retrain_weekly_days: int = RETRAIN_WEEKLY_DAYS,
    retrain_match_threshold: int = RETRAIN_MATCH_THRESHOLD,
) -> dict[str, Any]:
    state = _load_training_state(path)
    bootstrapped = False
    if not state:
        state = _build_bootstrap_training_state(tours)
        bootstrapped = True
        if write_bootstrap_state:
            _safe_json_dump(path, state)

    now = datetime.now(UTC)
    decisions: dict[str, Any] = {}
    tours_to_retrain: list[str] = []

    for tour in tours:
        current = _collect_feature_metadata(tour)
        stored = state.get("tours", {}).get(tour, {})
        missing_artifacts = _artifacts_missing(tour)

        if not current.get("exists", False):
            decisions[tour] = {
                "tour": tour,
                "trigger": False,
                "reason": "missing_features",
                "message": f"Feature file missing for {tour}.",
                "missing_artifacts": missing_artifacts,
            }
            continue

        current_rows = int(current.get("feature_rows", 0))
        stored_rows = int(stored.get("feature_rows", 0) or 0)
        new_rows = max(0, current_rows - stored_rows)

        current_date_max = current.get("feature_date_max")
        stored_date_max = stored.get("feature_date_max")
        has_newer_date = bool(current_date_max and (not stored_date_max or current_date_max > stored_date_max))
        has_new_data = new_rows > 0 or has_newer_date

        schema_changed = (
            bool(stored.get("feature_schema_signature"))
            and current.get("feature_schema_signature") != stored.get("feature_schema_signature")
        )
        trained_at = _parse_utc_timestamp(stored.get("trained_at"))
        weekly_due = bool(
            has_new_data
            and trained_at is not None
            and now - trained_at >= timedelta(days=retrain_weekly_days)
        )

        reason = "skip"
        trigger = False
        message = "No retraining needed."

        if missing_artifacts:
            reason = "missing_artifacts"
            trigger = True
            message = f"Model artifacts missing for {tour}: {', '.join(missing_artifacts)}."
        elif schema_changed:
            reason = "feature_schema_changed"
            trigger = True
            message = f"Feature schema changed for {tour}."
        elif new_rows >= retrain_match_threshold:
            reason = "new_matches_threshold"
            trigger = True
            message = f"{new_rows} new feature rows since last training for {tour}."
        elif weekly_due:
            reason = "weekly_new_data"
            trigger = True
            message = f"New data detected and weekly retrain window reached for {tour}."
        elif trained_at is None and not stored.get("bootstrapped", False):
            reason = "missing_trained_at"
            trigger = True
            message = f"Training timestamp missing for {tour}."
        elif bootstrapped:
            reason = "bootstrapped_state"
            message = f"Bootstrapped training state for {tour}; future runs will compare against this baseline."

        decision = {
            "tour": tour,
            "trigger": trigger,
            "reason": reason,
            "message": message,
            "trained_at": stored.get("trained_at"),
            "current_feature_rows": current_rows,
            "stored_feature_rows": stored_rows,
            "new_feature_rows": new_rows,
            "current_feature_date_max": current_date_max,
            "stored_feature_date_max": stored_date_max,
            "feature_schema_changed": schema_changed,
            "missing_artifacts": missing_artifacts,
            "bootstrapped_state": bootstrapped,
        }
        decisions[tour] = decision
        if trigger:
            tours_to_retrain.append(tour)

    return {
        "state_file": str(path),
        "bootstrapped_state": bootstrapped,
        "tours": decisions,
        "tours_to_retrain": tours_to_retrain,
        "triggered": bool(tours_to_retrain),
    }


def maybe_retrain_models(
    tours: tuple[str, ...] = ("atp", "wta"),
    *,
    path: Path = TRAINING_STATE_FILE,
    force: bool = False,
    write_bootstrap_state: bool = True,
    optuna_trials: int = OPTUNA_TRIALS,
    use_optuna: bool = False,
    fast: bool = False,
    max_rows: int | None = None,
    train_cutoff: str = TRAIN_CUTOFF,
    fallback_latest_months: int = 3,
    cv_folds: int = DEFAULT_TEMPORAL_CV_FOLDS,
) -> dict[str, Any]:
    policy = evaluate_retrain_policy(
        tours=tours,
        path=path,
        write_bootstrap_state=write_bootstrap_state,
    )
    selected_tours = list(tours) if force else list(policy.get("tours_to_retrain", []))
    if not selected_tours:
        if not force:
            summary: list[str] = []
            now = datetime.now(UTC)
            for tour in tours:
                decision = policy.get("tours", {}).get(tour, {})
                trained_at = _parse_utc_timestamp(decision.get("trained_at"))
                days_ago = "unknown"
                if trained_at is not None:
                    days_ago = str(max(0, int((now - trained_at).total_seconds() // 86400)))
                summary.append(
                    f"{str(tour).upper()}: last trained {days_ago}d ago, "
                    f"{int(decision.get('new_feature_rows', 0))} new matches "
                    f"(threshold: {RETRAIN_MATCH_THRESHOLD})"
                )
            if summary:
                log.info("Retrain skipped: %s", "; ".join(summary))
        return {
            "triggered": False,
            "force": force,
            "tours": [],
            "policy": policy,
            "message": "Retraining skipped by policy.",
        }

    training = train_models(
        tours=tuple(selected_tours),
        optuna_trials=optuna_trials,
        use_optuna=use_optuna,
        fast=fast,
        max_rows=max_rows,
        train_cutoff=train_cutoff,
        fallback_latest_months=fallback_latest_months,
        cv_folds=cv_folds,
    )
    return {
        "triggered": True,
        "force": force,
        "tours": selected_tours,
        "policy": policy,
        "training": training,
        "message": "Retraining completed.",
    }


def _split_temporal(
    df: pd.DataFrame,
    train_cutoff: str = TRAIN_CUTOFF,
    fallback_latest_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    cutoff = pd.Timestamp(train_cutoff)
    train_df = df[df["match_date"] < cutoff].copy()
    test_df = df[df["match_date"] >= cutoff].copy()

    if not train_df.empty and not test_df.empty:
        return train_df, test_df, "strict_cutoff"

    # Fallback: use latest N months as test window when strict cutoff has no split.
    if fallback_latest_months > 0 and not df.empty:
        max_date = df["match_date"].max()
        latest_start = max_date - pd.DateOffset(months=fallback_latest_months)
        train_df = df[df["match_date"] < latest_start].copy()
        test_df = df[df["match_date"] >= latest_start].copy()
        if not train_df.empty and not test_df.empty:
            return train_df, test_df, f"fallback_latest_{fallback_latest_months}m"

    # Last resort.
    split_idx = int(len(df) * 0.8)
    split_idx = max(1, min(split_idx, len(df) - 1))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df, "fallback_80_20_temporal"


def _split_train_validation(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(train_df) <= 1:
        return train_df.copy(), train_df.copy()

    if len(train_df) < 50:
        val_rows = min(10, max(1, len(train_df) // 5))
        return train_df.iloc[:-val_rows].copy(), train_df.iloc[-val_rows:].copy()

    idx = int(len(train_df) * 0.8)
    idx = max(1, min(idx, len(train_df) - 1))
    return train_df.iloc[:idx].copy(), train_df.iloc[idx:].copy()


def _choose_iteration_count(model: Any, fallback: int, attr_name: str) -> int:
    best_iter = getattr(model, attr_name, None)
    if best_iter is None:
        best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None:
        return int(fallback)
    return max(1, int(best_iter) + 1)


def _target_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "target_mean": None,
            "target_counts": {"0": 0, "1": 0},
            "date_min": None,
            "date_max": None,
        }

    counts = df[TARGET_COL].astype(int).value_counts().to_dict()
    return {
        "rows": int(len(df)),
        "target_mean": float(df[TARGET_COL].astype(float).mean()),
        "target_counts": {
            "0": int(counts.get(0, 0)),
            "1": int(counts.get(1, 0)),
        },
        "date_min": df["match_date"].min().strftime("%Y-%m-%d"),
        "date_max": df["match_date"].max().strftime("%Y-%m-%d"),
    }


def _prepare_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str]]:
    feature_cols = [c for c in train_df.columns if c not in set(META_COLS + [TARGET_COL])]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    x_train = train_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()

    for col in cat_cols:
        x_train[col] = x_train[col].astype("string").fillna("Unknown")
        x_test[col] = x_test[col].astype("string").fillna("Unknown")

    for col in num_cols:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce")

    medians = x_train[num_cols].median(numeric_only=True)
    x_train[num_cols] = x_train[num_cols].fillna(medians).fillna(0)
    x_test[num_cols] = x_test[num_cols].fillna(medians).fillna(0)

    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    return x_train, y_train, x_test, y_test, feature_cols, cat_cols


def _to_xgb_matrix(
    x_train: pd.DataFrame,
    x_eval: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_enc = pd.get_dummies(x_train, columns=cat_cols, dummy_na=True)
    eval_enc = pd.get_dummies(x_eval, columns=cat_cols, dummy_na=True)
    train_cols = list(train_enc.columns)
    eval_enc = eval_enc.reindex(columns=train_cols, fill_value=0)

    # XGBoost rejects feature names containing [, ], or <.
    seen: dict[str, int] = {}
    sanitized_cols: list[str] = []
    for col in train_cols:
        base = re.sub(r"[\[\]<>]", "_", str(col))
        base = re.sub(r"\s+", "_", base)
        base = re.sub(r"_+", "_", base).strip("_")
        if not base:
            base = "feature"

        count = seen.get(base, 0)
        name = base if count == 0 else f"{base}_{count}"
        seen[base] = count + 1
        sanitized_cols.append(name)

    train_enc.columns = sanitized_cols
    eval_enc.columns = sanitized_cols
    return train_enc, eval_enc


def _calibration_metrics(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(probs, bin_edges) - 1
    indices = np.clip(indices, 0, bins - 1)

    rows: list[dict[str, float]] = []
    ece = 0.0
    n = len(probs)

    for i in range(bins):
        mask = indices == i
        count = int(mask.sum())
        if count == 0:
            continue
        avg_pred = float(probs[mask].mean())
        avg_true = float(y_true[mask].mean())
        gap = abs(avg_true - avg_pred)
        ece += gap * (count / n)
        rows.append(
            {
                "bin": i,
                "count": count,
                "avg_pred": avg_pred,
                "avg_true": avg_true,
                "abs_gap": gap,
            }
        )

    return float(ece), rows


def _metric_pack(y_true: pd.Series, probs: np.ndarray) -> dict[str, float]:
    y_arr = y_true.to_numpy()
    preds = (probs >= 0.5).astype(int)
    ece, _ = _calibration_metrics(y_arr, probs)
    return {
        "accuracy": float(accuracy_score(y_arr, preds)),
        "log_loss": float(log_loss(y_arr, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y_arr, probs)),
        "ece": float(ece),
    }


def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _available_model_names() -> list[str]:
    available: list[str] = []
    if CatBoostClassifier is None:
        print("WARNING: CatBoost not installed; skipping catboost")
    else:
        available.append("catboost")

    if XGBClassifier is None:
        print("WARNING: XGBoost not installed; skipping xgboost")
    else:
        available.append("xgboost")

    if LGBMClassifier is None:
        print("WARNING: LightGBM not installed; skipping lgbm")
    else:
        available.append("lgbm")

    available.extend(["rf", "elasticnet", "logreg"])
    return available


def _ensemble_config_path(tour: str) -> Path:
    return MODELS_DIR / f"ensemble_config_{tour}.json"


def _load_ensemble_config(tour: str) -> dict[str, Any]:
    path = _ensemble_config_path(tour)
    if not path.exists():
        return dict(DEFAULT_ENSEMBLE_WEIGHTS)
    payload = _safe_json_load(path)
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        return dict(DEFAULT_ENSEMBLE_WEIGHTS)
    return payload


def _save_ensemble_config(
    tour: str,
    winner_name: str,
    winner_weights: dict[str, float],
    metrics: dict[str, float],
) -> Path:
    path = _ensemble_config_path(tour)
    _safe_json_dump(
        path,
        {
            "winner": winner_name,
            "weights": {name: float(weight) for name, weight in winner_weights.items()},
            "cv_log_loss": float(metrics["cv_log_loss"]),
            "cv_brier": float(metrics["cv_brier"]),
            "cv_accuracy": float(metrics["cv_accuracy"]),
            "selection_date": datetime.now(UTC).date().isoformat(),
        },
    )
    return path


def _default_model_params(model_name: str, *, fast: bool) -> dict[str, Any]:
    if model_name == "catboost":
        return {
            "iterations": 500 if fast else 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "border_count": 128,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "verbose": False,
            "random_seed": 42,
            "thread_count": -1,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": 500 if fast else 800,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
    if model_name == "lgbm":
        return {
            "n_estimators": 500 if fast else 800,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
    if model_name == "rf":
        return {
            "n_estimators": 300 if fast else 700,
            "max_depth": 10 if fast else None,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
    if model_name == "elasticnet":
        return {
            "solver": "saga",
            "l1_ratio": 0.5,
            "max_iter": 3000,
            "C": 1.0,
            "random_state": 42,
        }
    if model_name == "logreg":
        return {
            "solver": "lbfgs",
            "l1_ratio": 0,
            "max_iter": 3000,
            "C": 1.0,
            "random_state": 42,
        }
    raise KeyError(f"Unsupported model: {model_name}")


def _best_param_keys(model_name: str) -> set[str]:
    if model_name == "catboost":
        return {"learning_rate", "depth", "l2_leaf_reg", "iterations", "border_count"}
    if model_name in {"xgboost", "lgbm"}:
        return {
            "learning_rate",
            "max_depth",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
        }
    if model_name == "elasticnet":
        return {"C", "l1_ratio", "max_iter"}
    return set()


def _build_model_params(
    tour: str,
    model_name: str,
    *,
    fast: bool,
    tuned_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = _default_model_params(model_name, fast=fast)
    stored = _load_best_params(tour, model_name)
    allowed = _best_param_keys(model_name)
    if allowed:
        params.update({k: v for k, v in stored.items() if k in allowed})
    if tuned_params:
        params.update({k: v for k, v in tuned_params.items() if not allowed or k in allowed})
    if model_name == "catboost":
        params["thread_count"] = params.get("thread_count", -1)
    return params


def _build_model_matrices(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> dict[str, Any]:
    x_train, y_train, x_eval, y_eval, feature_cols, cat_cols = _prepare_frames(train_df, eval_df)
    num_cols = [c for c in feature_cols if c not in cat_cols]
    x_train_tab, x_eval_tab = _to_xgb_matrix(x_train, x_eval, cat_cols)
    return {
        "x_train_cat": x_train,
        "y_train": y_train,
        "x_eval_cat": x_eval,
        "y_eval": y_eval,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "x_train_tab": x_train_tab,
        "x_eval_tab": x_eval_tab,
    }


def _fit_selection_model(
    model_name: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if model_name not in {"catboost", "xgboost", "lgbm"}:
        return _serialize_params(params), {}

    train_core, val_core = _split_train_validation(train_df)
    if train_core.empty or val_core.empty:
        return _serialize_params(params), {}

    selection_inputs = _build_model_matrices(train_core, val_core)
    final_params = dict(params)
    details: dict[str, Any] = {}

    if model_name == "catboost":
        selection_model = CatBoostClassifier(**params)
        selection_model.fit(
            selection_inputs["x_train_cat"],
            selection_inputs["y_train"],
            cat_features=selection_inputs["cat_cols"],
            eval_set=(selection_inputs["x_eval_cat"], selection_inputs["y_eval"]),
            use_best_model=True,
            early_stopping_rounds=50,
        )
        best_iterations = _choose_iteration_count(selection_model, int(params["iterations"]), "best_iteration_")
        final_params["iterations"] = best_iterations
        details["best_iterations"] = int(best_iterations)
    elif model_name == "xgboost":
        selection_model = XGBClassifier(**{**params, "early_stopping_rounds": 50})
        selection_model.fit(
            selection_inputs["x_train_tab"],
            selection_inputs["y_train"],
            eval_set=[(selection_inputs["x_eval_tab"], selection_inputs["y_eval"])],
            verbose=False,
        )
        best_estimators = _choose_iteration_count(selection_model, int(params["n_estimators"]), "best_iteration")
        final_params["n_estimators"] = best_estimators
        details["best_estimators"] = int(best_estimators)
    else:
        selection_model = LGBMClassifier(**params)
        selection_model.fit(
            selection_inputs["x_train_tab"],
            selection_inputs["y_train"],
            eval_set=[(selection_inputs["x_eval_tab"], selection_inputs["y_eval"])],
            callbacks=[lgbm_early_stopping(50, verbose=False)] if lgbm_early_stopping is not None else None,
        )
        best_estimators = _choose_iteration_count(selection_model, int(params["n_estimators"]), "best_iteration_")
        final_params["n_estimators"] = best_estimators
        details["best_estimators"] = int(best_estimators)

    return _serialize_params(final_params), details


def _fit_model(
    model_name: str,
    params: dict[str, Any],
    matrices: dict[str, Any],
) -> Any:
    if model_name == "catboost":
        model = CatBoostClassifier(**params)
        model.fit(
            matrices["x_train_cat"],
            matrices["y_train"],
            cat_features=matrices["cat_cols"],
            verbose=False,
        )
        return model
    if model_name == "xgboost":
        model = XGBClassifier(**params)
        model.fit(matrices["x_train_tab"], matrices["y_train"], verbose=False)
        return model
    if model_name == "lgbm":
        model = LGBMClassifier(**params)
        model.fit(matrices["x_train_tab"], matrices["y_train"])
        return model
    if model_name == "rf":
        model = RandomForestClassifier(**params)
        model.fit(matrices["x_train_tab"], matrices["y_train"])
        return model
    if model_name in {"elasticnet", "logreg"}:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**params)),
            ]
        )
        model.fit(matrices["x_train_tab"], matrices["y_train"])
        return model
    raise KeyError(f"Unsupported model: {model_name}")


def _predict_model_proba(model_name: str, model: Any, matrices: dict[str, Any]) -> np.ndarray:
    if model_name == "catboost":
        return model.predict_proba(matrices["x_eval_cat"])[:, 1]
    return model.predict_proba(matrices["x_eval_tab"])[:, 1]


def _model_artifact_path(tour: str, model_name: str) -> Path:
    suffixes = {
        "catboost": f"catboost_{tour}.cbm",
        "xgboost": f"xgboost_{tour}.json",
        "lgbm": f"lgbm_{tour}.txt",
        "rf": f"rf_{tour}.pkl",
        "elasticnet": f"elasticnet_{tour}.pkl",
        "logreg": f"logreg_{tour}.pkl",
    }
    return MODELS_DIR / suffixes[model_name]


def _save_model_artifact(model_name: str, model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "catboost":
        model.save_model(path)
        return
    if model_name == "xgboost":
        model.save_model(path)
        return
    if model_name == "lgbm":
        booster = model.booster_ if hasattr(model, "booster_") else model
        with tempfile.NamedTemporaryFile(prefix="tennisbet_lgbm_", suffix=".txt", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            booster.save_model(str(temp_path))
            shutil.copyfile(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        return
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def _feature_importance_rows(model_name: str, model: Any, feature_names: list[str]) -> pd.DataFrame:
    if model_name == "catboost":
        values = list(model.get_feature_importance())
    else:
        estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
        if hasattr(estimator, "feature_importances_"):
            values = list(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            coef = np.ravel(getattr(estimator, "coef_"))
            values = list(np.abs(coef))
        else:
            return pd.DataFrame(columns=["model", "feature", "importance"])
    return pd.DataFrame({"model": model_name, "feature": feature_names, "importance": values})


def _mean_metric_from_rows(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _build_comparison_row(name: str, rows: list[dict[str, Any]], weights: dict[str, float] | None = None) -> dict[str, Any]:
    payload = {
        "name": name,
        "cv_log_loss": _mean_metric_from_rows(rows, "log_loss"),
        "cv_brier": _mean_metric_from_rows(rows, "brier"),
        "cv_accuracy": _mean_metric_from_rows(rows, "accuracy"),
    }
    if weights:
        payload["weights"] = {model_name: float(weight) for model_name, weight in weights.items()}
    return payload


def _print_model_comparison(rows: list[dict[str, Any]], winner_name: str) -> None:
    print("Model              CV Log-Loss   CV Brier   CV Accuracy")
    print("-------------------------------------------------------")
    for row in rows:
        marker = "  "
        if row["name"] == winner_name:
            marker = "* "
        print(
            f"{row['name']:<18}{row['cv_log_loss']:.3f}        "
            f"{row['cv_brier']:.3f}       {row['cv_accuracy']:.3f} {marker}".rstrip()
        )


def _confidence_from_probability_map(probability_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not probability_map:
        return np.array([], dtype=float), np.array([], dtype=object)
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
    return 1.0 - agreement_gap, confidence


def _build_tuning_folds(train_df: pd.DataFrame) -> list[dict[str, Any]]:
    folds = _build_temporal_cv_folds(
        train_df,
        target_folds=TUNING_FOLDS,
        min_train_rows=max(300, min(2000, max(300, len(train_df) // 4))),
        min_test_rows=max(75, min(300, max(75, len(train_df) // 12))),
    )
    if folds:
        return folds

    train_core, val_core = _split_train_validation(train_df)
    if train_core.empty or val_core.empty:
        return []
    return [
        {
            "fold": 1,
            "train_df": train_core,
            "test_df": val_core,
            "train_rows": int(len(train_core)),
            "test_rows": int(len(val_core)),
            "train_date_min": train_core["match_date"].min().strftime("%Y-%m-%d"),
            "train_date_max": train_core["match_date"].max().strftime("%Y-%m-%d"),
            "test_date_min": val_core["match_date"].min().strftime("%Y-%m-%d"),
            "test_date_max": val_core["match_date"].max().strftime("%Y-%m-%d"),
        }
    ]


def _create_optuna_study(saved_params: dict[str, Any]) -> Any:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )
    if saved_params:
        study.enqueue_trial(saved_params)
    return study


def _fit_optuna_catboost(
    *,
    tour: str,
    train_df: pd.DataFrame,
    n_trials: int,
    fast: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if optuna is None or CatBoostClassifier is None or n_trials <= 0:
        return {}, {}

    tuning_folds = _build_tuning_folds(train_df)
    if not tuning_folds:
        return {}, {}

    saved_params = _load_best_params(tour, "catboost")
    allowed_keys = {"learning_rate", "depth", "l2_leaf_reg", "iterations", "border_count"}
    queued_params = {k: v for k, v in saved_params.items() if k in allowed_keys}

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 500 if fast else 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": 42,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "verbose": False,
            # CatBoost sklearn wrapper exposes thread_count, not n_jobs.
            "thread_count": 1,
        }

        fold_losses: list[float] = []
        for step, fold in enumerate(tuning_folds, start=1):
            x_fold_train, y_fold_train, x_fold_val, y_fold_val, _, cat_cols = _prepare_frames(
                fold["train_df"],
                fold["test_df"],
            )
            model = CatBoostClassifier(**params)
            model.fit(
                x_fold_train,
                y_fold_train,
                cat_features=cat_cols,
                eval_set=(x_fold_val, y_fold_val),
                use_best_model=True,
                early_stopping_rounds=50,
            )
            preds = model.predict_proba(x_fold_val)[:, 1]
            fold_loss = float(log_loss(y_fold_val, preds, labels=[0, 1]))
            fold_losses.append(fold_loss)
            trial.report(float(np.mean(fold_losses)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    study = _create_optuna_study(queued_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return {}, {}

    best_trial = study.best_trial
    best_params = {k: v for k, v in best_trial.params.items() if k in allowed_keys}
    path = _save_best_params(tour, "catboost", best_trial.value, best_params)
    print(
        f"[Optuna][{tour.upper()}][catboost] best log_loss={best_trial.value:.6f} "
        f"params={json.dumps(best_params, sort_keys=True)} saved={path}"
    )
    return best_params, {
        "score": float(best_trial.value),
        "params": best_params,
        "path": str(path),
        "trials": int(len(study.trials)),
    }


def _fit_optuna_xgboost(
    *,
    tour: str,
    train_df: pd.DataFrame,
    n_trials: int,
    fast: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if optuna is None or XGBClassifier is None or n_trials <= 0:
        return {}, {}

    tuning_folds = _build_tuning_folds(train_df)
    if not tuning_folds:
        return {}, {}

    saved_params = _load_best_params(tour, "xgboost")
    allowed_keys = {
        "learning_rate",
        "max_depth",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
    }
    queued_params = {k: v for k, v in saved_params.items() if k in allowed_keys}

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300 if fast else 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }

        fold_losses: list[float] = []
        for step, fold in enumerate(tuning_folds, start=1):
            x_fold_train, y_fold_train, x_fold_val, y_fold_val, _, cat_cols = _prepare_frames(
                fold["train_df"],
                fold["test_df"],
            )
            x_fold_train_xgb, x_fold_val_xgb = _to_xgb_matrix(x_fold_train, x_fold_val, cat_cols)
            model = XGBClassifier(**{**params, "early_stopping_rounds": 50})
            model.fit(
                x_fold_train_xgb,
                y_fold_train,
                eval_set=[(x_fold_val_xgb, y_fold_val)],
                verbose=False,
            )
            preds = model.predict_proba(x_fold_val_xgb)[:, 1]
            fold_loss = float(log_loss(y_fold_val, preds, labels=[0, 1]))
            fold_losses.append(fold_loss)
            trial.report(float(np.mean(fold_losses)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    study = _create_optuna_study(queued_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return {}, {}

    best_trial = study.best_trial
    best_params = {k: v for k, v in best_trial.params.items() if k in allowed_keys}
    path = _save_best_params(tour, "xgboost", best_trial.value, best_params)
    print(
        f"[Optuna][{tour.upper()}][xgboost] best log_loss={best_trial.value:.6f} "
        f"params={json.dumps(best_params, sort_keys=True)} saved={path}"
    )
    return best_params, {
        "score": float(best_trial.value),
        "params": best_params,
        "path": str(path),
        "trials": int(len(study.trials)),
    }


def _fit_optuna_lightgbm(
    *,
    tour: str,
    train_df: pd.DataFrame,
    n_trials: int,
    fast: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if optuna is None or LGBMClassifier is None or n_trials <= 0:
        return {}, {}

    tuning_folds = _build_tuning_folds(train_df)
    if not tuning_folds:
        return {}, {}

    saved_params = _load_best_params(tour, "lgbm")
    allowed_keys = {
        "learning_rate",
        "max_depth",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
    }
    queued_params = {k: v for k, v in saved_params.items() if k in allowed_keys}

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300 if fast else 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

        fold_losses: list[float] = []
        for step, fold in enumerate(tuning_folds, start=1):
            matrices = _build_model_matrices(fold["train_df"], fold["test_df"])
            model = LGBMClassifier(**params)
            fit_kwargs: dict[str, Any] = {}
            if lgbm_early_stopping is not None:
                fit_kwargs["callbacks"] = [lgbm_early_stopping(50, verbose=False)]
            model.fit(
                matrices["x_train_tab"],
                matrices["y_train"],
                eval_set=[(matrices["x_eval_tab"], matrices["y_eval"])],
                **fit_kwargs,
            )
            preds = model.predict_proba(matrices["x_eval_tab"])[:, 1]
            fold_loss = float(log_loss(matrices["y_eval"], preds, labels=[0, 1]))
            fold_losses.append(fold_loss)
            trial.report(float(np.mean(fold_losses)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    study = _create_optuna_study(queued_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return {}, {}

    best_trial = study.best_trial
    best_params = {k: v for k, v in best_trial.params.items() if k in allowed_keys}
    path = _save_best_params(tour, "lgbm", best_trial.value, best_params)
    print(
        f"[Optuna][{tour.upper()}][lgbm] best log_loss={best_trial.value:.6f} "
        f"params={json.dumps(best_params, sort_keys=True)} saved={path}"
    )
    return best_params, {
        "score": float(best_trial.value),
        "params": best_params,
        "path": str(path),
        "trials": int(len(study.trials)),
    }


def _build_comparison_folds(df: pd.DataFrame, target_folds: int) -> list[dict[str, Any]]:
    folds = _build_temporal_cv_folds(
        df,
        target_folds=target_folds,
        min_train_rows=max(300, min(2000, max(300, len(df) // 4))),
        min_test_rows=max(75, min(300, max(75, len(df) // 12))),
    )
    if folds:
        return folds

    train_core, val_core = _split_train_validation(df)
    if train_core.empty or val_core.empty:
        return []
    return [
        {
            "fold": 1,
            "train_df": train_core,
            "test_df": val_core,
            "train_rows": int(len(train_core)),
            "test_rows": int(len(val_core)),
            "train_date_min": train_core["match_date"].min().strftime("%Y-%m-%d"),
            "train_date_max": train_core["match_date"].max().strftime("%Y-%m-%d"),
            "test_date_min": val_core["match_date"].min().strftime("%Y-%m-%d"),
            "test_date_max": val_core["match_date"].max().strftime("%Y-%m-%d"),
        }
    ]


def _run_temporal_cv(
    tour: str,
    train_df: pd.DataFrame,
    *,
    model_params: dict[str, dict[str, Any]],
    target_folds: int,
) -> dict[str, Any]:
    folds = _build_comparison_folds(train_df, target_folds=target_folds)
    if not folds:
        return {
            "enabled": False,
            "requested_folds": int(target_folds),
            "completed_folds": 0,
            "message": "Insufficient data for temporal CV.",
            "folds": [],
            "summary": {},
            "comparison_rows": [],
            "winner_row": None,
        }

    base_rows: dict[str, list[dict[str, Any]]] = {model_name: [] for model_name in model_params}
    fold_predictions: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []

    for fold in folds:
        matrices = _build_model_matrices(fold["train_df"], fold["test_df"])
        y_eval = matrices["y_eval"]
        per_model_probs: dict[str, np.ndarray] = {}
        fold_row: dict[str, Any] = {
            "fold": fold["fold"],
            "train_rows": fold["train_rows"],
            "test_rows": fold["test_rows"],
            "train_date_min": fold["train_date_min"],
            "train_date_max": fold["train_date_max"],
            "test_date_min": fold["test_date_min"],
            "test_date_max": fold["test_date_max"],
        }

        for model_name, params in model_params.items():
            final_params, _ = _fit_selection_model(model_name, params, fold["train_df"])
            model = _fit_model(model_name, final_params, matrices)
            probs = _predict_model_proba(model_name, model, matrices)
            metrics = _metric_pack(y_eval, probs)
            per_model_probs[model_name] = probs
            base_rows[model_name].append(metrics)
            fold_row[f"{model_name}_log_loss"] = metrics["log_loss"]
            fold_row[f"{model_name}_brier"] = metrics["brier"]
            fold_row[f"{model_name}_accuracy"] = metrics["accuracy"]

        fold_predictions.append({"y_true": y_eval.to_numpy(), "probabilities": per_model_probs})
        fold_metadata.append(fold_row)

    comparison_rows = [_build_comparison_row(model_name, rows) for model_name, rows in base_rows.items()]
    ranked_models = [row["name"] for row in sorted(comparison_rows, key=lambda row: row["cv_log_loss"])]

    ensemble_candidates: list[tuple[str, dict[str, float]]] = []
    if {"catboost", "xgboost"}.issubset(model_params):
        ensemble_candidates.append(("cat85_xgb15", {"catboost": 0.85, "xgboost": 0.15}))
        ensemble_candidates.append(("cat60_xgb40", {"catboost": 3 / 5, "xgboost": 2 / 5}))
    if {"catboost", "lgbm"}.issubset(model_params):
        ensemble_candidates.append(("cat70_lgbm30", {"catboost": 0.70, "lgbm": 0.30}))
    if len(ranked_models) >= 2:
        ensemble_candidates.append(("top2_equal", {name: 0.5 for name in ranked_models[:2]}))
    if len(ranked_models) >= 3:
        ensemble_candidates.append(("top3_equal", {name: 1.0 / 3.0 for name in ranked_models[:3]}))

    for candidate_name, weights in ensemble_candidates:
        candidate_rows: list[dict[str, Any]] = []
        for idx, fold_data in enumerate(fold_predictions):
            ensemble_probs = np.zeros_like(next(iter(fold_data["probabilities"].values())), dtype=float)
            for model_name, weight in weights.items():
                ensemble_probs += float(weight) * fold_data["probabilities"][model_name]
            metrics = _metric_pack(pd.Series(fold_data["y_true"]), ensemble_probs)
            candidate_rows.append(metrics)
            fold_metadata[idx][f"{candidate_name}_log_loss"] = metrics["log_loss"]
            fold_metadata[idx][f"{candidate_name}_brier"] = metrics["brier"]
            fold_metadata[idx][f"{candidate_name}_accuracy"] = metrics["accuracy"]
        comparison_rows.append(_build_comparison_row(candidate_name, candidate_rows, weights=weights))

    comparison_rows = sorted(comparison_rows, key=lambda row: row["cv_log_loss"])
    winner_row = comparison_rows[0]

    summary: dict[str, Any] = {}
    folds_df = pd.DataFrame(fold_metadata)
    for row in comparison_rows:
        name = row["name"]
        for metric_name in ("log_loss", "brier", "accuracy"):
            col = f"{name}_{metric_name}"
            if col in folds_df.columns:
                summary[f"{col}_mean"] = float(folds_df[col].mean())
                summary[f"{col}_std"] = float(folds_df[col].std(ddof=0))

    _print_model_comparison(comparison_rows, winner_row["name"])
    print(
        f"[CV][{tour.upper()}] winner={winner_row['name']} "
        f"log_loss={winner_row['cv_log_loss']:.6f} "
        f"weights={json.dumps(winner_row.get('weights', {winner_row['name']: 1.0}), sort_keys=True)}"
    )

    return {
        "enabled": True,
        "requested_folds": int(target_folds),
        "completed_folds": int(len(fold_metadata)),
        "message": "Temporal CV completed.",
        "folds": fold_metadata,
        "summary": summary,
        "comparison_rows": comparison_rows,
        "winner_row": winner_row,
    }


def _train_and_score_split(
    tour: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    optuna_trials: int,
    use_optuna: bool,
    fast: bool,
    cv_folds: int = DEFAULT_TEMPORAL_CV_FOLDS,
) -> dict[str, Any]:
    available_models = _available_model_names()
    if not available_models:
        raise RuntimeError("No supported models are available for training.")

    tuning_report: dict[str, Any] = {"catboost": None, "xgboost": None, "lgbm": None}
    tuned_params: dict[str, dict[str, Any]] = {}
    train_core, val_core = _split_train_validation(train_df)
    should_tune = use_optuna and optuna is not None and optuna_trials > 0 and len(train_core) > 500 and len(val_core) > 100

    if should_tune and "catboost" in available_models:
        tuned_params["catboost"], tuning_report["catboost"] = _fit_optuna_catboost(
            tour=tour,
            train_df=train_df,
            n_trials=optuna_trials,
            fast=fast,
        )
    if should_tune and "xgboost" in available_models:
        tuned_params["xgboost"], tuning_report["xgboost"] = _fit_optuna_xgboost(
            tour=tour,
            train_df=train_df,
            n_trials=optuna_trials,
            fast=fast,
        )
    if should_tune and "lgbm" in available_models:
        tuned_params["lgbm"], tuning_report["lgbm"] = _fit_optuna_lightgbm(
            tour=tour,
            train_df=train_df,
            n_trials=optuna_trials,
            fast=fast,
        )

    model_params = {
        model_name: _build_model_params(
            tour,
            model_name,
            fast=fast,
            tuned_params=tuned_params.get(model_name),
        )
        for model_name in available_models
    }
    temporal_cv = _run_temporal_cv(
        tour,
        train_df,
        model_params=model_params,
        target_folds=cv_folds,
    )
    winner_row = temporal_cv.get("winner_row")
    if winner_row is None:
        fallback_model = available_models[0]
        winner_row = {
            "name": fallback_model,
            "weights": {fallback_model: 1.0},
            "cv_log_loss": math.nan,
            "cv_brier": math.nan,
            "cv_accuracy": math.nan,
        }
    winner_weights = dict(winner_row.get("weights") or {winner_row["name"]: 1.0})

    matrices = _build_model_matrices(train_df, test_df)
    final_models: dict[str, Any] = {}
    final_params: dict[str, dict[str, Any]] = {}
    selection_details: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}

    for model_name in available_models:
        resolved_params, resolved_selection = _fit_selection_model(model_name, model_params[model_name], train_df)
        model = _fit_model(model_name, resolved_params, matrices)
        probs = _predict_model_proba(model_name, model, matrices)
        final_models[model_name] = model
        final_params[model_name] = resolved_params
        selection_details[model_name] = resolved_selection
        probabilities[model_name] = probs
        metrics[model_name] = _metric_pack(matrices["y_eval"], probs)

    ensemble_probs = np.zeros_like(next(iter(probabilities.values())), dtype=float)
    for model_name, weight in winner_weights.items():
        if model_name not in probabilities:
            continue
        ensemble_probs += float(weight) * probabilities[model_name]
    calibration_ece, calibration_rows = _calibration_metrics(matrices["y_eval"].to_numpy(), ensemble_probs)
    metrics["ensemble"] = _metric_pack(matrices["y_eval"], ensemble_probs)
    metrics["ensemble_calibration_ece"] = calibration_ece

    return {
        "models": final_models,
        "feature_cols": matrices["feature_cols"],
        "cat_cols": matrices["cat_cols"],
        "num_cols": matrices["num_cols"],
        "x_train": matrices["x_train_cat"],
        "x_test": matrices["x_eval_cat"],
        "x_train_xgb": matrices["x_train_tab"],
        "x_test_xgb": matrices["x_eval_tab"],
        "y_test": matrices["y_eval"],
        "probabilities": {
            **probabilities,
            "ensemble": ensemble_probs,
        },
        "metrics": metrics,
        "calibration_rows": calibration_rows,
        "params": final_params,
        "selection": selection_details,
        "tuning": tuning_report,
        "split_summary": {
            "train": _target_summary(train_df),
            "validation": _target_summary(val_core),
            "test": _target_summary(test_df),
        },
        "temporal_cv": temporal_cv,
        "model_comparison": temporal_cv.get("comparison_rows", []),
        "ensemble_config": {
            "winner": winner_row["name"],
            "weights": winner_weights,
            "cv_log_loss": float(winner_row.get("cv_log_loss", math.nan)),
            "cv_brier": float(winner_row.get("cv_brier", math.nan)),
            "cv_accuracy": float(winner_row.get("cv_accuracy", math.nan)),
        },
    }


def _build_temporal_cv_folds(
    df: pd.DataFrame,
    target_folds: int,
    min_train_rows: int = 2000,
    min_test_rows: int = 200,
) -> list[dict[str, Any]]:
    if target_folds <= 0 or len(df) < (min_train_rows + min_test_rows):
        return []

    test_rows = max(min_test_rows, len(df) // (target_folds + 4))
    train_rows = max(min_train_rows, test_rows * 4)
    max_folds = (len(df) - train_rows) // test_rows
    fold_count = min(target_folds, max_folds)
    if fold_count <= 0:
        return []

    start_offset = max(0, len(df) - (train_rows + fold_count * test_rows))
    folds: list[dict[str, Any]] = []
    for idx in range(fold_count):
        train_start = start_offset + (idx * test_rows)
        train_end = train_start + train_rows
        test_end = train_end + test_rows
        train_split = df.iloc[train_start:train_end].copy()
        test_split = df.iloc[train_end:test_end].copy()
        if len(train_split) < min_train_rows or len(test_split) < min_test_rows:
            continue
        folds.append(
            {
                "fold": idx + 1,
                "train_df": train_split,
                "test_df": test_split,
                "train_rows": int(len(train_split)),
                "test_rows": int(len(test_split)),
                "train_date_min": train_split["match_date"].min().strftime("%Y-%m-%d"),
                "train_date_max": train_split["match_date"].max().strftime("%Y-%m-%d"),
                "test_date_min": test_split["match_date"].min().strftime("%Y-%m-%d"),
                "test_date_max": test_split["match_date"].max().strftime("%Y-%m-%d"),
            }
        )
    return folds


def train_for_tour(
    tour: str,
    optuna_trials: int = OPTUNA_TRIALS,
    use_optuna: bool = False,
    fast: bool = False,
    max_rows: int | None = None,
    train_cutoff: str = TRAIN_CUTOFF,
    fallback_latest_months: int = 3,
    cv_folds: int = DEFAULT_TEMPORAL_CV_FOLDS,
) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_features(tour)
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:].copy()

    train_df, test_df, split_mode = _split_temporal(
        df,
        train_cutoff=train_cutoff,
        fallback_latest_months=fallback_latest_months,
    )
    split_result = _train_and_score_split(
        tour=tour,
        train_df=train_df,
        test_df=test_df,
        optuna_trials=optuna_trials,
        use_optuna=use_optuna,
        fast=fast,
        cv_folds=cv_folds,
    )
    temporal_cv = split_result["temporal_cv"]

    feature_cols = split_result["feature_cols"]
    cat_cols = split_result["cat_cols"]
    num_cols = split_result["num_cols"]
    x_train = split_result["x_train"]
    x_train_xgb = split_result["x_train_xgb"]
    ensemble_probs = split_result["probabilities"]["ensemble"]
    calibration_rows = split_result["calibration_rows"]
    probabilities = split_result["probabilities"]
    final_params = split_result["params"]
    ensemble_config = split_result["ensemble_config"]
    model_comparison = split_result["model_comparison"]
    tuning_report = split_result.get("tuning", {})

    preprocess_payload = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "numeric_medians": {
            col: float(x_train[col].median()) if pd.notna(x_train[col].median()) else 0.0
            for col in num_cols
        },
        "xgb_feature_cols": list(x_train_xgb.columns),
    }
    preprocess_path = MODELS_DIR / f"preprocess_{tour}.json"
    _safe_json_dump(preprocess_path, preprocess_payload)
    metrics_ens = split_result["metrics"]["ensemble"]
    calibration_ece = split_result["metrics"]["ensemble_calibration_ece"]
    calibration_path = MODELS_DIR / f"calibration_{tour}.csv"
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)
    temporal_cv_path = MODELS_DIR / f"temporal_cv_{tour}.csv"
    pd.DataFrame(temporal_cv.get("folds", [])).to_csv(temporal_cv_path, index=False)
    ensemble_config_path = _save_ensemble_config(
        tour,
        ensemble_config["winner"],
        ensemble_config["weights"],
        {
            "cv_log_loss": ensemble_config["cv_log_loss"],
            "cv_brier": ensemble_config["cv_brier"],
            "cv_accuracy": ensemble_config["cv_accuracy"],
        },
    )

    artifact_paths: dict[str, str] = {
        "preprocess": str(preprocess_path),
        "calibration": str(calibration_path),
        "temporal_cv": str(temporal_cv_path),
        "ensemble_config": str(ensemble_config_path),
    }
    importance_frames: list[pd.DataFrame] = []
    for model_name, model in split_result["models"].items():
        model_path = _model_artifact_path(tour, model_name)
        _save_model_artifact(model_name, model, model_path)
        artifact_paths[f"{model_name}_model"] = str(model_path)
        feature_names = feature_cols if model_name == "catboost" else list(x_train_xgb.columns)
        importance = _feature_importance_rows(model_name, model, feature_names)
        if not importance.empty:
            importance_frames.append(importance)

    fi_path = MODELS_DIR / f"feature_importance_{tour}.csv"
    if importance_frames:
        pd.concat(importance_frames, ignore_index=True).sort_values("importance", ascending=False).to_csv(fi_path, index=False)
    else:
        pd.DataFrame(columns=["model", "feature", "importance"]).to_csv(fi_path, index=False)
    artifact_paths["feature_importance"] = str(fi_path)

    pred_path = MODELS_DIR / f"test_predictions_{tour}.csv"
    pred_df = test_df[["match_key", "match_date", "p1_name", "p2_name", TARGET_COL]].copy()
    for model_name in BASE_MODEL_ORDER:
        pred_df[f"{model_name}_prob"] = probabilities.get(model_name, np.nan)
    pred_df["ensemble_prob"] = ensemble_probs
    interval_payload = _build_prediction_interval_payload(split_result["y_test"], ensemble_probs)
    interval_radius = float(interval_payload.get("residual_quantile", 0.0))
    pred_df["ensemble_prob_lower"] = np.clip(ensemble_probs - interval_radius, 0.0, 1.0)
    pred_df["ensemble_prob_upper"] = np.clip(ensemble_probs + interval_radius, 0.0, 1.0)
    pred_df["ensemble_interval_width"] = pred_df["ensemble_prob_upper"] - pred_df["ensemble_prob_lower"]
    pred_df.to_csv(pred_path, index=False)
    artifact_paths["predictions"] = str(pred_path)
    uncertainty_path = MODELS_DIR / f"uncertainty_{tour}.json"
    _safe_json_dump(uncertainty_path, interval_payload)
    artifact_paths["uncertainty"] = str(uncertainty_path)

    report = {
        "tour": tour,
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_cutoff": train_cutoff,
        "split_mode": split_mode,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "split_summary": split_result["split_summary"],
        "optuna_trials_requested": int(optuna_trials) if use_optuna else 0,
        "optuna_trials_used": int(optuna_trials) if use_optuna and optuna is not None else 0,
        "catboost_params": final_params.get("catboost"),
        "xgboost_params": final_params.get("xgboost"),
        "lgbm_params": final_params.get("lgbm"),
        "rf_params": final_params.get("rf"),
        "elasticnet_params": final_params.get("elasticnet"),
        "logreg_params": final_params.get("logreg"),
        "selection": split_result["selection"],
        "best_params_files": {
            "catboost": str(_best_params_path(tour, "catboost")),
            "xgboost": str(_best_params_path(tour, "xgboost")),
            "lgbm": str(_best_params_path(tour, "lgbm")),
        },
        "tuning": tuning_report,
        "metrics": split_result["metrics"],
        "prediction_interval": interval_payload,
        "model_comparison": model_comparison,
        "ensemble_winner": ensemble_config["winner"],
        "ensemble_config": ensemble_config,
        "temporal_cv": {
            **temporal_cv,
            "cv_metrics_file": str(temporal_cv_path),
        },
        "artifacts": artifact_paths,
    }

    report_path = MODELS_DIR / f"model_report_{tour}.json"
    _safe_json_dump(report_path, report)

    return report


def train_models(
    tours: tuple[str, ...] = ("atp", "wta"),
    optuna_trials: int = OPTUNA_TRIALS,
    use_optuna: bool = False,
    fast: bool = False,
    max_rows: int | None = None,
    train_cutoff: str = TRAIN_CUTOFF,
    fallback_latest_months: int = 3,
    cv_folds: int = DEFAULT_TEMPORAL_CV_FOLDS,
) -> dict[str, Any]:
    results = {
        tour: train_for_tour(
            tour=tour,
            optuna_trials=optuna_trials,
            use_optuna=use_optuna,
            fast=fast,
            max_rows=max_rows,
            train_cutoff=train_cutoff,
            fallback_latest_months=fallback_latest_months,
            cv_folds=cv_folds,
        )
        for tour in tours
    }

    rows = []
    for tour, rep in results.items():
        winner_name = rep.get("ensemble_winner")
        rows.append(
            {
                "trained_at": rep["trained_at"],
                "tour": tour,
                "train_cutoff": rep.get("train_cutoff"),
                "split_mode": rep["split_mode"],
                "rows_train": rep["rows_train"],
                "rows_test": rep["rows_test"],
                "catboost_accuracy": rep["metrics"].get("catboost", {}).get("accuracy"),
                "catboost_log_loss": rep["metrics"].get("catboost", {}).get("log_loss"),
                "xgboost_accuracy": rep["metrics"].get("xgboost", {}).get("accuracy"),
                "xgboost_log_loss": rep["metrics"].get("xgboost", {}).get("log_loss"),
                "lgbm_accuracy": rep["metrics"].get("lgbm", {}).get("accuracy"),
                "lgbm_log_loss": rep["metrics"].get("lgbm", {}).get("log_loss"),
                "rf_accuracy": rep["metrics"].get("rf", {}).get("accuracy"),
                "rf_log_loss": rep["metrics"].get("rf", {}).get("log_loss"),
                "elasticnet_accuracy": rep["metrics"].get("elasticnet", {}).get("accuracy"),
                "elasticnet_log_loss": rep["metrics"].get("elasticnet", {}).get("log_loss"),
                "logreg_accuracy": rep["metrics"].get("logreg", {}).get("accuracy"),
                "logreg_log_loss": rep["metrics"].get("logreg", {}).get("log_loss"),
                "ensemble_accuracy": rep["metrics"]["ensemble"]["accuracy"],
                "ensemble_log_loss": rep["metrics"]["ensemble"]["log_loss"],
                "ensemble_ece": rep["metrics"]["ensemble_calibration_ece"],
                "interval_confidence_level": rep.get("prediction_interval", {}).get("confidence_level"),
                "interval_empirical_coverage": rep.get("prediction_interval", {}).get("empirical_coverage"),
                "interval_mean_width": rep.get("prediction_interval", {}).get("mean_interval_width"),
                "ensemble_winner": winner_name,
                "temporal_cv_folds": rep.get("temporal_cv", {}).get("completed_folds", 0),
                "temporal_cv_winner_accuracy_mean": rep.get("temporal_cv", {}).get("summary", {}).get(f"{winner_name}_accuracy_mean") if winner_name else None,
                "temporal_cv_winner_log_loss_mean": rep.get("temporal_cv", {}).get("summary", {}).get(f"{winner_name}_log_loss_mean") if winner_name else None,
                "temporal_cv_winner_brier_mean": rep.get("temporal_cv", {}).get("summary", {}).get(f"{winner_name}_brier_mean") if winner_name else None,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_path = MODELS_DIR / "model_metrics.csv"
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        old = pd.read_csv(metrics_path)
        metrics_df = pd.concat([old, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)

    state = _safe_json_load(LAST_UPDATE_FILE)
    state["model_last_trained"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    state["model_training"] = {
        "tours": list(tours),
        "train_cutoff": train_cutoff,
        "fallback_latest_months": fallback_latest_months,
        "cv_folds": cv_folds,
        "report_files": {tour: str(MODELS_DIR / f"model_report_{tour}.json") for tour in tours},
        "metrics_file": str(metrics_path),
    }
    _safe_json_dump(LAST_UPDATE_FILE, state)
    training_state = _save_training_state_after_training(
        results,
        train_cutoff=train_cutoff,
        fallback_latest_months=fallback_latest_months,
        cv_folds=cv_folds,
    )

    return {
        "results": results,
        "metrics_file": str(metrics_path),
        "training_state_file": str(TRAINING_STATE_FILE),
        "training_state": training_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TennisBet multi-model ensemble")
    parser.add_argument("--tours", nargs="+", default=["atp", "wta"], help="Tours to train, e.g. --tours atp wta")
    parser.add_argument("--tune", action="store_true", help="Run full Optuna tuning before final training")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip optuna tuning")
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--fast", action="store_true", help="Use faster settings for iteration/testing")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved training configuration without fitting models")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on most recent rows per tour")
    parser.add_argument("--train-cutoff", type=str, default=TRAIN_CUTOFF, help="Temporal split cutoff (YYYY-MM-DD)")
    parser.add_argument(
        "--fallback-latest-months",
        type=int,
        default=3,
        help="If strict cutoff has no train/test split, use latest N months as test window",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_TEMPORAL_CV_FOLDS,
        help="Number of rolling temporal CV folds to run; use 0 to disable",
    )

    args = parser.parse_args()
    tours = tuple(
        part.strip()
        for raw in args.tours
        for part in str(raw).split(",")
        if part.strip()
    )
    use_optuna = bool(args.tune and not args.skip_optuna)
    trial_count = 5 if args.fast and args.optuna_trials > 5 else args.optuna_trials

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "tours": list(tours),
                    "tune": bool(args.tune),
                    "use_optuna": use_optuna,
                    "optuna_trials": int(trial_count),
                    "fast": bool(args.fast),
                    "max_rows": args.max_rows,
                    "train_cutoff": args.train_cutoff,
                    "fallback_latest_months": args.fallback_latest_months,
                    "cv_folds": args.cv_folds,
                    "stored_best_params": {
                        tour: {
                            "catboost": _load_best_params(tour, "catboost"),
                            "xgboost": _load_best_params(tour, "xgboost"),
                            "lgbm": _load_best_params(tour, "lgbm"),
                        }
                        for tour in tours
                    },
                    "stored_ensemble_configs": {tour: _load_ensemble_config(tour) for tour in tours},
                },
                indent=2,
            )
        )
        return

    report = train_models(
        tours=tours,
        optuna_trials=trial_count,
        use_optuna=use_optuna,
        fast=args.fast,
        max_rows=args.max_rows,
        train_cutoff=args.train_cutoff,
        fallback_latest_months=args.fallback_latest_months,
        cv_folds=args.cv_folds,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
