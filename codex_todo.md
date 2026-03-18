# CODEX TODO - TennisBet v2

## Completed (Phases 1-6)
- [x] Data foundation: Sackmann repos, custom matches, Flashscore scraper, data pipeline, ELO engine
- [x] Feature engineering: 56 features, leakage audit, deterministic P1/P2 ordering
- [x] Model training: CatBoost + XGBoost ensemble (60/40), Optuna smoke-tested
- [x] Prediction & value engine: edge detection, bankroll allocation, backtest framework
- [x] Streamlit app: 3 tabs, sidebar controls, manual entry, bankroll tracking
- [x] GitHub Actions: daily email workflow, `.env.example`, SMTP integration
- [x] Staleness fix: `update_data_sources()` now considers processed + custom match dates (not just raw Sackmann)
- [x] Capital fix: `DEFAULT_CAPITAL` updated to EUR 100, `bankroll_log.json` aligned

---

## Phase 7: Quick Wins (< 1 hour each)

### 7.1 Dynamic train cutoff
- [x] Replace hardcoded `TRAIN_CUTOFF = "2025-01-01"` in `config.py` with a rolling window
- Use: `TRAIN_CUTOFF = (date.today() - timedelta(days=180)).strftime("%Y-%m-%d")`
- Impact: without this, the test set keeps growing and training data stays frozen
- Files: `config.py`, verify `model_training.py` still respects it

### 7.2 Overround adjustment in value engine
- [x] Normalize implied probabilities for bookmaker margin in `_compute_edges()`
- Currently: `fair_implied_p1 = (1/odds_p1) / (1/odds_p1 + 1/odds_p2)` -- this is already normalized but named misleadingly
- Real fix: apply power method or multiplicative removal to get true fair odds
- The current normalization underestimates edge slightly (minor but free to fix)
- Files: `src/value_engine.py` lines 118-138

### 7.3 Remove plaintext secrets from repo
- [x] Delete `GMAIL_secrets.txt` from the repo and add to `.gitignore`
- Already have `.env` mechanism; the secrets file is redundant and a security risk
- Files: `GMAIL_secrets.txt`, `.gitignore`

### 7.4 Skip email when zero bets
- [x] Add a gate in `daily_report.py`: if `recommendations.empty` and no errors, skip sending email
- Reduces inbox noise on days with no actionable bets
- Could add a `--skip-empty` flag to `daily_report.py` CLI args
- Files: `src/daily_report.py` around line 581

- [x] Add `--bankroll` CLI arg to `daily_report.py` (like the UFC project has)
- Currently reads from `bankroll_log.json` only; explicit override per run is cleaner
- Also update `generate_recommendations()` default capital from `5.0` to `DEFAULT_CAPITAL`
- Files: `src/daily_report.py`, `src/value_engine.py` line 205

---

## Phase 8: Model & Prediction Hardening (half-day each)

### 8.1 Temporal cross-validation
- [x] Replace single train/test split with sliding-window temporal CV
- The UFC project does this: train on window, validate on next period, repeat
- Gives more robust accuracy estimates; single splits can be misleading
- Files: `src/model_training.py`

### 8.2 Retrain policy (event-driven)
- [x] Implement automatic retrain triggers (adapted from UFC project):
  - Retrain weekly if new matches landed since last training
  - Retrain immediately if 50+ new matches landed (tennis has more volume than UFC)
  - Retrain immediately if feature schema changes
  - Skip if nothing changed
- Store last training metadata in `models/training_state.json`
- Files: `src/model_training.py`, `src/daily_report.py`

### 8.3 Run full Optuna tuning
- [ ] Execute 50-trial Optuna for both CatBoost and XGBoost, both tours
- Currently only smoke-tested with reduced trials
- Prerequisite: close the 2025 data gap first (see 8.5)
- Files: `src/model_training.py`

### 8.4 Odds name matching with alias table
- [x] Create `data/meta/player_aliases.json` mapping common name variants
- Current fuzzy matching silently drops matches when names don't align
- The UFC project solves this with a fighter aliases table in the canonical DB
- [x] Add logging when a prediction has no odds match (to catch silent drops)
- Files: `src/value_engine.py`, `src/predictor.py`

### 8.5 Close the 2025 raw data gap
- [ ] Sackmann `*_matches_2025.csv` / `*_matches_2026.csv` return HTTP 404
- Need alternative source or more aggressive Flashscore historical scraping
- Without this, `TRAIN_CUTOFF=2025-01-01` yields a tiny custom-only test window
- This blocks meaningful backtesting and full Optuna tuning

---

## Phase 9: Architecture Upgrades (1-3 days each)

### 9.1 SQLite canonical DB (from UFC project)
- [ ] Replace CSV-based storage with a normalized SQLite database
- Entities: `players`, `player_aliases`, `tournaments`, `matches`, `odds`, `predictions`, `bankroll_log`
- Benefits: proper upserts, no duplicate rows, incremental ingestion with cursors, faster queries
- The UFC project's `db/ufc.sqlite` pattern is the reference implementation
- This is the single biggest architectural improvement available
- Files: new `db/tennis.sqlite`, refactor `src/data_pipeline.py`, `src/data_updater.py`

### 9.2 Multi-bookmaker odds aggregation
- [ ] Support multiple odds sources with fair probability computed as median across bookmakers
- UFC uses median of 2+ bookmakers + min bookmaker count filter
- Currently tennis takes whatever single odds source is available
- Would materially improve edge estimation reliability
- Files: `src/odds_scraper.py`, `src/value_engine.py`

### 9.3 Proper Kelly Criterion
- [x] Replace simplified edge-based allocation with fractional Kelly
- Current sizing: `base_pct + edge_bonus * confidence_mult` (heuristic)
- Kelly: `f* = (bp - q) / b` where b=odds-1, p=model_prob, q=1-p
- Use fractional Kelly (e.g. 0.25x) for safety
- Files: `src/value_engine.py`

### 9.4 Live backtest on rolling window
- [x] Allow backtesting on arbitrary date ranges with regenerated predictions
- Current backtest loads static `test_predictions_{tour}.csv` from training time
- Need: select date range -> rebuild features as-of each date -> predict -> compare to actuals
- Files: `src/backtest.py`

---

## Phase 10: Advanced (1+ week each)

### 10.1 Prediction intervals / uncertainty quantification
- [ ] Output prediction intervals instead of just point estimates
- Options: conformal prediction, bootstrap ensembles, or Bayesian approaches
- Helps filter bets where the model is uncertain even if the point estimate looks good
- Files: `src/predictor.py`, `src/model_training.py`

### 10.2 Feature expansion: serve/return stats
- [ ] Integrate detailed serve/return statistics from Match Charting Project or similar
- Current serve features (ace%, 1st serve%, BP save%) are rolling-20 averages
- Surface-specific serve metrics would add signal (especially for grass vs clay)
- Files: `src/feature_engineering.py`

### 10.3 In-play odds movement tracking
- [ ] Track odds movement between scraping windows
- Steam moves (sharp money) are a strong signal the model can incorporate
- Requires more frequent odds snapshots (e.g. every 2 hours on match day)
- Files: `src/odds_scraper.py`, new `src/odds_tracker.py`

### 10.4 Automated performance dashboard
- [ ] Weekly email/HTML summary of model performance: actual ROI, calibration drift, hit rate by confidence tier
- Catch model degradation early instead of discovering it after losses
- Files: `src/daily_report.py` or new `src/performance_report.py`

---

## Bug Tracker

### Known bugs
- [x] `generate_recommendations()` default capital is `5.0` (line 205 of value_engine.py) -- should be `DEFAULT_CAPITAL`
- [x] ~~Streamlit app double-logging~~ -- verified: `_log_recommendations_to_prediction_log()` in app.py has deduplication via `existing_ids` set
- [ ] `bankroll_log.json` has UTF-8 BOM encoding issue (causes `json.load()` to fail without `utf-8-sig`)
- [x] Custom match entry in app.py accepts arbitrary player IDs without validation against known players

### Resolved bugs
- [x] Staleness calculation used only raw Sackmann dates, ignoring processed/custom matches (fixed: now takes max of all sources)
- [x] Capital hardcoded at EUR 5.00 (fixed: updated to EUR 100.00 in config + bankroll_log)
