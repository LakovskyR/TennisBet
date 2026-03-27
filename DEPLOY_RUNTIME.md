# Deploy Runtime Files

This repository should stay focused on the files required for the deployed
Streamlit app to start, refresh data, and generate predictions.

Keep tracked for deploy:

- `app.py`, `config.py`, `src/`, `requirements.txt`, `packages.txt`
- `.streamlit/secrets.toml.example`
- `publish_runtime_update.bat` for local runtime refresh + push
- runtime model artifacts:
  - `models/catboost_*.cbm`
  - `models/xgboost_*.json`
  - `models/lgbm_*.txt`
  - `models/preprocess_*.json`
  - `models/ensemble_config_*.json`
  - `models/uncertainty_*.json`
- lightweight model metadata used by the UI:
  - `models/model_metrics.csv`
  - `models/calibration_*.csv`
  - `models/feature_importance_*.csv`
  - `models/model_report_*.json`
- runtime data required by the app:
  - `data/custom/*.csv`
  - `data/meta/last_update.json`
  - `data/meta/player_aliases.json`
  - `data/processed/*_elo_*.csv`
  - `data/processed/*_matches_master.csv`
  - `data/processed/*_player_features.csv`

Do not track for deploy:

- local automation scripts such as `retrain_weekly.bat`
- raw historical archives under `data/raw/`
- training diagnostics and backtest exports
- optimizer parameter snapshots
- local prompt files and scratch notes

Notes:

- The deployed app still requires the runtime inference artifacts listed above.
- Removing `catboost`, `xgboost`, or `lgbm` runtime files will break prediction generation.
- The app tolerates missing backtest and training-diagnostic CSVs by showing empty views.
