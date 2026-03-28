# TennisBet

TennisBet is a local-only tennis match prediction and value-betting assistant. It uses historical ATP/WTA match data, incremental pipeline/ELO/features, pre-trained models, live odds scraping, and a Streamlit UI for daily betting review.

## Workflow

Daily use:
- run `streamlit run app.py`
- click `Refresh All`
- app refreshes match data, ELO, features, odds, and predictions
- app uses the existing model artifacts under `models/`

Weekly maintenance:
- run `retrain_weekly.bat`
- script refreshes data, rebuilds pipeline/ELO/features, and retrains models locally
- no GitHub, Streamlit Cloud, or push step is required

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

## Main files

- `app.py`: daily Streamlit interface
- `retrain_weekly.bat`: weekly local rebuild + retrain flow
- `src/`: pipeline, scraper, ELO, feature engineering, training, prediction, value engine
- `data/raw/`, `data/processed/`, `data/odds/`, `data/custom/`, `data/meta/`
- `models/`: saved model artifacts and metadata
- `codex_todo.md`: current task tracker

## Common commands

```bash
streamlit run app.py
python -m src.data_updater
python -m src.data_pipeline
python -m src.elo_engine
python -m src.feature_engineering
python -m src.model_training --tours atp wta --tune
python -m src.backtest
```

## Notes

- Predictions are pre-match only.
- If live odds scraping fails, you can upload an odds CSV in the app.
- If model files are stale or invalid, rerun `retrain_weekly.bat`.
- This is a statistical tool, not financial advice.
