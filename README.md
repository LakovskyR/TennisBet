# TennisBet

TennisBet is a local-first tennis match prediction and value-betting assistant built around:

- Sackmann ATP/WTA historical data
- incremental pipeline + ELO ratings
- CatBoost/XGBoost ensemble match predictions
- odds-driven value detection and bankroll tracking
- a Streamlit app for daily recommendations

## Current status

Implemented now:
- data updater with git/HTTP fallback and staleness tracking
- pipeline, ELO engine, feature engineering, model training, predictor, value engine, backtest
- Streamlit app with:
  - data refresh, odds refresh, prediction refresh
  - manual odds CSV upload fallback
  - custom recent match form + custom CSV upload
  - recommendation cards with adjustable threshold and max-bets controls
  - model performance charts and data status tab
  - prediction log + bankroll settlement flow
- headless `src/daily_report.py` cycle with HTML email generation and Gmail-secrets fallback for local runs
- GitHub Actions workflow scaffold for scheduled daily runs

Still pending:
- broader error handling and unified logging across modules
- optional workflow modernization for future GitHub Actions runtime changes
- inbox-level confirmation of the live email formatting

## Project layout

- `app.py`
- `config.py`
- `src/`
- `data/raw/`, `data/processed/`, `data/odds/`, `data/custom/`, `data/meta/`
- `models/`
- `codex_todo.md`, `claude_todo.md`

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Optional CLI runs:

```bash
python -m src.data_updater
python -m src.data_pipeline
python -m src.elo_engine
python -m src.model_training
python -m src.backtest
```

## Notes

- Predictions are for pre-match analysis only.
- If live scraping fails, upload an odds CSV in the Streamlit sidebar.
- This is a statistical tool for educational use, not financial advice.
