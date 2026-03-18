from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

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
    import optuna
except Exception:  # pragma: no cover
    optuna = None

CAT_COLS = ["surface", "tournament_level", "round", "best_of"]
TARGET_COL = "p1_wins"
META_COLS = ["match_key", "match_date", "tour", "p1_id", "p2_id", "p1_name", "p2_name"]
DEFAULT_TEMPORAL_CV_FOLDS = 3
TRAINING_STATE_FILE = MODELS_DIR / "training_state.json"
RETRAIN_WEEKLY_DAYS = 7
RETRAIN_MATCH_THRESHOLD = 50


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
        "preprocess": MODELS_DIR / f"preprocess_{tour}.json",
        "report": MODELS_DIR / f"model_report_{tour}.json",
    }


def _artifacts_missing(tour: str) -> list[str]:
    return [name for name, path in _artifacts_for_tour(tour).items() if not path.exists()]


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


def _fit_optuna_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list[str],
    n_trials: int,
    fast: bool,
) -> dict[str, Any]:
    if optuna is None or CatBoostClassifier is None or n_trials <= 0:
        return {}

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "iterations": 400 if fast else 800,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed": 42,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "early_stopping_rounds": 50,
            "verbose": False,
        }
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, cat_features=cat_cols, eval_set=(x_val, y_val), use_best_model=True)
        preds = model.predict_proba(x_val)[:, 1]
        return log_loss(y_val, preds, labels=[0, 1])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params if study.best_trial else {}


def _fit_optuna_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int,
    fast: bool,
) -> dict[str, Any]:
    if optuna is None or XGBClassifier is None or n_trials <= 0:
        return {}

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": 400 if fast else 800,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }
        model = XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        preds = model.predict_proba(x_val)[:, 1]
        return log_loss(y_val, preds, labels=[0, 1])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params if study.best_trial else {}


def _train_and_score_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    optuna_trials: int,
    use_optuna: bool,
    fast: bool,
) -> dict[str, Any]:
    train_core, val_core = _split_train_validation(train_df)

    x_train, y_train, x_test, y_test, feature_cols, cat_cols = _prepare_frames(train_df, test_df)
    x_train_core, y_train_core, x_val_core, y_val_core, _, _ = _prepare_frames(train_core, val_core)
    num_cols = [c for c in feature_cols if c not in cat_cols]

    cat_params = {
        "iterations": 500 if fast else 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "early_stopping_rounds": 50,
        "verbose": False,
        "random_seed": 42,
    }

    xgb_params = {
        "n_estimators": 500 if fast else 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "reg_lambda": 3,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
        "early_stopping_rounds": 50,
    }

    tuned_cat: dict[str, Any] = {}
    tuned_xgb: dict[str, Any] = {}

    if use_optuna and optuna is not None and optuna_trials > 0 and len(train_core) > 500 and len(val_core) > 100:
        tuned_cat = _fit_optuna_catboost(
            x_train_core,
            y_train_core,
            x_val_core,
            y_val_core,
            cat_cols,
            optuna_trials,
            fast,
        )

        x_train_core_xgb, x_val_core_xgb = _to_xgb_matrix(x_train_core, x_val_core, cat_cols)
        tuned_xgb = _fit_optuna_xgboost(
            x_train_core_xgb,
            y_train_core,
            x_val_core_xgb,
            y_val_core,
            optuna_trials,
            fast,
        )

    cat_params.update({k: v for k, v in tuned_cat.items() if k in {"learning_rate", "depth", "l2_leaf_reg"}})
    xgb_params.update(
        {
            k: v
            for k, v in tuned_xgb.items()
            if k in {"learning_rate", "max_depth", "reg_lambda", "subsample", "colsample_bytree"}
        }
    )

    cat_selection_model = CatBoostClassifier(**cat_params)
    cat_selection_model.fit(
        x_train_core,
        y_train_core,
        cat_features=cat_cols,
        eval_set=(x_val_core, y_val_core),
        use_best_model=True,
    )
    cat_final_iterations = _choose_iteration_count(cat_selection_model, cat_params["iterations"], "best_iteration_")
    cat_final_params = {k: v for k, v in cat_params.items() if k != "early_stopping_rounds"}
    cat_final_params["iterations"] = cat_final_iterations

    cat_model = CatBoostClassifier(**cat_final_params)
    cat_model.fit(x_train, y_train, cat_features=cat_cols, verbose=False)

    x_train_core_xgb, x_val_core_xgb = _to_xgb_matrix(x_train_core, x_val_core, cat_cols)
    xgb_selection_model = XGBClassifier(**xgb_params)
    xgb_selection_model.fit(x_train_core_xgb, y_train_core, eval_set=[(x_val_core_xgb, y_val_core)], verbose=False)
    xgb_final_estimators = _choose_iteration_count(
        xgb_selection_model,
        xgb_params["n_estimators"],
        "best_iteration",
    )

    x_train_xgb, x_test_xgb = _to_xgb_matrix(x_train, x_test, cat_cols)
    xgb_final_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
    xgb_final_params["n_estimators"] = xgb_final_estimators
    xgb_model = XGBClassifier(**xgb_final_params)
    xgb_model.fit(x_train_xgb, y_train, verbose=False)

    cat_probs = cat_model.predict_proba(x_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(x_test_xgb)[:, 1]
    ensemble_probs = 0.6 * cat_probs + 0.4 * xgb_probs
    calibration_ece, calibration_rows = _calibration_metrics(y_test.to_numpy(), ensemble_probs)

    return {
        "models": {
            "catboost": cat_model,
            "xgboost": xgb_model,
        },
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "x_train": x_train,
        "x_test": x_test,
        "x_train_xgb": x_train_xgb,
        "x_test_xgb": x_test_xgb,
        "y_test": y_test,
        "probabilities": {
            "catboost": cat_probs,
            "xgboost": xgb_probs,
            "ensemble": ensemble_probs,
        },
        "metrics": {
            "catboost": _metric_pack(y_test, cat_probs),
            "xgboost": _metric_pack(y_test, xgb_probs),
            "ensemble": _metric_pack(y_test, ensemble_probs),
            "ensemble_calibration_ece": calibration_ece,
        },
        "calibration_rows": calibration_rows,
        "params": {
            "catboost": cat_final_params,
            "xgboost": xgb_final_params,
        },
        "selection": {
            "catboost_best_iterations": int(cat_final_iterations),
            "xgboost_best_iterations": int(xgb_final_estimators),
        },
        "split_summary": {
            "train": _target_summary(train_df),
            "validation": _target_summary(val_core),
            "test": _target_summary(test_df),
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


def _run_temporal_cv(
    df: pd.DataFrame,
    *,
    fast: bool,
    target_folds: int,
) -> dict[str, Any]:
    folds = _build_temporal_cv_folds(df, target_folds=target_folds)
    if not folds:
        return {
            "enabled": False,
            "requested_folds": int(target_folds),
            "completed_folds": 0,
            "message": "Insufficient data for temporal CV.",
            "folds": [],
            "summary": {},
        }

    rows: list[dict[str, Any]] = []
    for fold in folds:
        result = _train_and_score_split(
            fold["train_df"],
            fold["test_df"],
            optuna_trials=0,
            use_optuna=False,
            fast=fast,
        )
        rows.append(
            {
                "fold": fold["fold"],
                "train_rows": fold["train_rows"],
                "test_rows": fold["test_rows"],
                "train_date_min": fold["train_date_min"],
                "train_date_max": fold["train_date_max"],
                "test_date_min": fold["test_date_min"],
                "test_date_max": fold["test_date_max"],
                "catboost_accuracy": result["metrics"]["catboost"]["accuracy"],
                "catboost_log_loss": result["metrics"]["catboost"]["log_loss"],
                "xgboost_accuracy": result["metrics"]["xgboost"]["accuracy"],
                "xgboost_log_loss": result["metrics"]["xgboost"]["log_loss"],
                "ensemble_accuracy": result["metrics"]["ensemble"]["accuracy"],
                "ensemble_log_loss": result["metrics"]["ensemble"]["log_loss"],
                "ensemble_brier": result["metrics"]["ensemble"]["brier"],
                "ensemble_ece": result["metrics"]["ensemble"]["ece"],
            }
        )

    folds_df = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for metric_col in [
        "catboost_accuracy",
        "catboost_log_loss",
        "xgboost_accuracy",
        "xgboost_log_loss",
        "ensemble_accuracy",
        "ensemble_log_loss",
        "ensemble_brier",
        "ensemble_ece",
    ]:
        summary[f"{metric_col}_mean"] = float(folds_df[metric_col].mean())
        summary[f"{metric_col}_std"] = float(folds_df[metric_col].std(ddof=0))

    return {
        "enabled": True,
        "requested_folds": int(target_folds),
        "completed_folds": int(len(rows)),
        "message": "Temporal CV completed.",
        "folds": rows,
        "summary": summary,
    }


def train_for_tour(
    tour: str,
    optuna_trials: int = OPTUNA_TRIALS,
    use_optuna: bool = True,
    fast: bool = False,
    max_rows: int | None = None,
    train_cutoff: str = TRAIN_CUTOFF,
    fallback_latest_months: int = 3,
    cv_folds: int = DEFAULT_TEMPORAL_CV_FOLDS,
) -> dict[str, Any]:
    if CatBoostClassifier is None or XGBClassifier is None:
        raise RuntimeError(
            "CatBoost and/or XGBoost are not installed. Run: python -m pip install catboost xgboost optuna"
        )

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
        train_df,
        test_df,
        optuna_trials=optuna_trials,
        use_optuna=use_optuna,
        fast=fast,
    )
    temporal_cv = _run_temporal_cv(df, fast=fast, target_folds=cv_folds)

    cat_model = split_result["models"]["catboost"]
    xgb_model = split_result["models"]["xgboost"]
    feature_cols = split_result["feature_cols"]
    cat_cols = split_result["cat_cols"]
    num_cols = split_result["num_cols"]
    x_train = split_result["x_train"]
    x_train_xgb = split_result["x_train_xgb"]
    y_test = split_result["y_test"]
    cat_probs = split_result["probabilities"]["catboost"]
    xgb_probs = split_result["probabilities"]["xgboost"]
    ensemble_probs = split_result["probabilities"]["ensemble"]
    calibration_rows = split_result["calibration_rows"]
    cat_final_params = split_result["params"]["catboost"]
    xgb_final_params = split_result["params"]["xgboost"]
    cat_final_iterations = split_result["selection"]["catboost_best_iterations"]
    xgb_final_estimators = split_result["selection"]["xgboost_best_iterations"]

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
    metrics_cat = split_result["metrics"]["catboost"]
    metrics_xgb = split_result["metrics"]["xgboost"]
    metrics_ens = split_result["metrics"]["ensemble"]
    calibration_ece = split_result["metrics"]["ensemble_calibration_ece"]
    calibration_path = MODELS_DIR / f"calibration_{tour}.csv"
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)
    temporal_cv_path = MODELS_DIR / f"temporal_cv_{tour}.csv"
    pd.DataFrame(temporal_cv.get("folds", [])).to_csv(temporal_cv_path, index=False)

    cat_path = MODELS_DIR / f"catboost_{tour}.cbm"
    xgb_path = MODELS_DIR / f"xgboost_{tour}.json"
    cat_model.save_model(cat_path)
    xgb_model.save_model(xgb_path)

    cat_importance = pd.DataFrame(
        {
            "model": "catboost",
            "feature": feature_cols,
            "importance": cat_model.get_feature_importance(),
        }
    )

    xgb_importance = pd.DataFrame(
        {
            "model": "xgboost",
            "feature": x_train_xgb.columns,
            "importance": xgb_model.feature_importances_,
        }
    )

    fi_path = MODELS_DIR / f"feature_importance_{tour}.csv"
    pd.concat([cat_importance, xgb_importance], ignore_index=True).sort_values(
        "importance", ascending=False
    ).to_csv(fi_path, index=False)

    pred_path = MODELS_DIR / f"test_predictions_{tour}.csv"
    pred_df = test_df[["match_key", "match_date", "p1_name", "p2_name", TARGET_COL]].copy()
    pred_df["catboost_prob"] = cat_probs
    pred_df["xgboost_prob"] = xgb_probs
    pred_df["ensemble_prob"] = ensemble_probs
    pred_df.to_csv(pred_path, index=False)

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
        "catboost_params": cat_final_params,
        "xgboost_params": xgb_final_params,
        "selection": split_result["selection"],
        "metrics": {
            "catboost": metrics_cat,
            "xgboost": metrics_xgb,
            "ensemble": metrics_ens,
            "ensemble_calibration_ece": calibration_ece,
        },
        "temporal_cv": {
            **temporal_cv,
            "cv_metrics_file": str(temporal_cv_path),
        },
        "artifacts": {
            "catboost_model": str(cat_path),
            "xgboost_model": str(xgb_path),
            "preprocess": str(preprocess_path),
            "feature_importance": str(fi_path),
            "calibration": str(calibration_path),
            "temporal_cv": str(temporal_cv_path),
            "predictions": str(pred_path),
        },
    }

    report_path = MODELS_DIR / f"model_report_{tour}.json"
    _safe_json_dump(report_path, report)

    return report


def train_models(
    tours: tuple[str, ...] = ("atp", "wta"),
    optuna_trials: int = OPTUNA_TRIALS,
    use_optuna: bool = True,
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
        rows.append(
            {
                "trained_at": rep["trained_at"],
                "tour": tour,
                "train_cutoff": rep.get("train_cutoff"),
                "split_mode": rep["split_mode"],
                "rows_train": rep["rows_train"],
                "rows_test": rep["rows_test"],
                "catboost_accuracy": rep["metrics"]["catboost"]["accuracy"],
                "catboost_log_loss": rep["metrics"]["catboost"]["log_loss"],
                "xgboost_accuracy": rep["metrics"]["xgboost"]["accuracy"],
                "xgboost_log_loss": rep["metrics"]["xgboost"]["log_loss"],
                "ensemble_accuracy": rep["metrics"]["ensemble"]["accuracy"],
                "ensemble_log_loss": rep["metrics"]["ensemble"]["log_loss"],
                "ensemble_ece": rep["metrics"]["ensemble_calibration_ece"],
                "temporal_cv_folds": rep.get("temporal_cv", {}).get("completed_folds", 0),
                "temporal_cv_ensemble_accuracy_mean": rep.get("temporal_cv", {}).get("summary", {}).get("ensemble_accuracy_mean"),
                "temporal_cv_ensemble_log_loss_mean": rep.get("temporal_cv", {}).get("summary", {}).get("ensemble_log_loss_mean"),
                "temporal_cv_ensemble_ece_mean": rep.get("temporal_cv", {}).get("summary", {}).get("ensemble_ece_mean"),
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
    parser = argparse.ArgumentParser(description="Train TennisBet CatBoost/XGBoost models")
    parser.add_argument("--tours", default="atp,wta", help="Comma-separated tours, e.g. atp,wta")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip optuna tuning")
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--fast", action="store_true", help="Use faster settings for iteration/testing")
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
    tours = tuple(t.strip() for t in args.tours.split(",") if t.strip())
    use_optuna = not args.skip_optuna
    trial_count = 5 if args.fast and args.optuna_trials > 5 else args.optuna_trials

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
