# CODEX TODO - TennisBet Build

## Phase 1: Data Foundation
- [x] Clone/download JeffSackmann/tennis_atp into `data/raw/tennis_atp/`
- [x] Clone/download JeffSackmann/tennis_wta into `data/raw/tennis_wta/`
- [x] Create `data/custom/`, `data/meta/`, `data/odds/` directories with empty starter files
- [x] Build `config.py` with all constants
- [x] Build `data_updater.py` - git pull primary, HTTP fallback, staleness check per Data Lifecycle spec
- [x] Implement custom match entry: Streamlit form + CSV upload, saved to `data/custom/`
- [x] Implement Flashscore results scraper (completed match results as tertiary data source)
- [x] Build `data_pipeline.py` - full implementation per Module 1 spec, merge custom matches
- [x] Run pipeline, validate outputs (duplicates/date order checks passed)
- [x] Build `elo_engine.py` - full implementation per Module 2 spec (incremental-capable)
- [x] Run ELO computation, validate chronological ordering
- [x] Initialize `data/meta/last_update.json` with first run metadata

## Phase 2: Feature Engineering & Models
- [x] Build `feature_engineering.py` - all feature groups per Module 3 spec
- [x] Build `tournament_country.json` mapping (tournament name -> ISO country code)
- [x] Validate player IOC codes from Sackmann `player.csv` for home country features (`ioc_coverage` ATP/WTA > 99.9%)
- [x] Validate no data leakage (spot-check 10 random matches via `data/meta/leakage_audit_{tour}.json`)
- [x] Build `model_training.py` - CatBoost + XGBoost per Module 4 spec
- [ ] Run Optuna hyperparameter tuning (50 trials each model, both tours) *(deferred for now per priority instructions; smoke-tested on reduced trials)*
- [x] Train models, save to `models/` directory (`catboost_{tour}.cbm`, `xgboost_{tour}.json`)
- [x] Log test set metrics: accuracy, log-loss, calibration (`models/model_metrics.csv`, `models/calibration_{tour}.csv`)

## Phase 3: Prediction & Value Engine
- [x] Build `predictor.py` - ensemble logic per Module 5 spec
- [x] Build `odds_scraper.py` - Flashscore Selenium scraper + fallback/manual CSV path
- [x] Build `value_engine.py` - value detection + allocation per Module 7 spec
- [x] Build `backtest.py` - 3-month backtest framework per Module 8 spec
- [x] Run backtest, save results (`models/backtest_results.csv`; currently no odds overlap to compute ROI)

## Phase 4: Streamlit App
- [x] Build `app.py` - all 3 tabs per Streamlit spec
- [x] Wire up sidebar controls (capital, tour filter, manual custom CSV upload)
- [x] Implement "Update Data" button in sidebar (triggers updater + pipeline + ELO)
- [x] Implement staleness warning banner (color-coded)
- [x] Implement "Add Recent Match" custom entry form in sidebar
- [x] Implement "Refresh odds" flow
- [x] Implement bankroll tracking: result input after bet day -> updates `bankroll_log.json`
- [x] Implement `prediction_log.csv`: log every prediction + actual outcome for performance tracking
- [x] Test full end-to-end: update -> scrape -> predict -> recommend -> display
- [x] Build `requirements.txt` with pinned versions

## Phase 5: Polish
- [ ] Add error handling throughout (network failures, missing data, scraper blocks)
- [ ] Add logging (Python logging module, file + console) *(currently partial logging in scraper only)*
- [x] Write `README.md` with setup instructions
- [x] Final test with real upcoming matches

## Phase 6: GitHub Repo + Daily Email Automation
- [x] Initialize git repo in `bet/` folder, create `.gitignore` (exclude `data/raw/`, `models/*.cbm`, `__pycache__`, `.env`)
- [x] Create private GitHub repo and push initial codebase
- [x] Create `src/daily_report.py` - headless script (no Streamlit) for full cycle + SMTP email
- [x] Create `.github/workflows/daily_bet.yml` (cron + workflow_dispatch + artifact upload)
- [x] Create `.env.example` documenting required secrets
- [x] Test workflow with `workflow_dispatch`
- [ ] Verify email arrives with correct formatting and recommendations
