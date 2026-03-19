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
- [x] Execute 50-trial Optuna for both CatBoost and XGBoost, both tours
- Currently only smoke-tested with reduced trials
- Prerequisite: close the 2025 data gap first (see 8.5)
- Files: `src/model_training.py`
- extended: multi-model comparison added

### 8.4 Odds name matching with alias table
- [x] Create `data/meta/player_aliases.json` mapping common name variants
- Current fuzzy matching silently drops matches when names don't align
- The UFC project solves this with a fighter aliases table in the canonical DB
- [x] Add logging when a prediction has no odds match (to catch silent drops)
- Files: `src/value_engine.py`, `src/predictor.py`

### 8.5 Close the 2025 raw data gap
- [x] Sackmann `*_matches_2025.csv` / `*_matches_2026.csv` return HTTP 404
- Need alternative source or more aggressive Flashscore historical scraping
- Without this, `TRAIN_CUTOFF=2025-01-01` yields a tiny custom-only test window
- This blocks meaningful backtesting and full Optuna tuning

---

## Phase 8.6: WTA Backfill — Fix Tournament Slugs + Retrain
- [x] Fix 9 tournament slugs in `src/wta_backfill.py` (Tennis Explorer uses `-wta` suffix for WTA pages):
  - `mutua-madrid-open` → `madrid-wta`
  - `internazionali-bnl-d-italia` → `rome-wta`
  - `national-bank-open` → `montreal-wta`
  - `western-and-southern-open` → `cincinnati-wta`
  - `abu-dhabi` → `abu-dhabi-wta` (both 2025 and 2026 registries)
  - `seoul` → `seoul-wta`
  - `singapore` → `singapore-wta`
  - `austin` → `austin-wta`
  - `bad-homburg` → `bad-homburg-wta`
  - Remove San Diego entirely (not on Tennis Explorer)
- [x] Fix `rtrvr_scrape()` in `wta_backfill.py`: rtrvr.ai returns data in `tree` field, not `content`. Use `tab.get("content") or tab.get("tree")` as fallback.
- [x] Delete old WTA CSVs and re-run: `python -m src.wta_backfill --years 2025 2026`
- [x] Retrain after backfill: follow `codex_retrain_prompt.md`
- Expected: ~1,800-2,200 matches for 2025 (up from 1,337), ~480-550 for 2026
- Full details: see `deepseek_wta_backfill_prompt.md`

### 8.7 Clean up project root — move unused files to `archive/`
- [x] Create `archive/` folder and move the following files into it:

**Debug scripts (one-off, no longer needed):**
  - `check_madrid_html.py`
  - `debug_madrid.py`
  - `debug_tournaments.py`
  - `debug_wta_adelaide.py`
  - `debug_wta_parser.py`
  - `debug_wta_rounds.py`
  - `discover_slugs.py`
  - `discover_slugs_robust.py`
  - `validate_wta.py`

**Debug output files:**
  - `debug_ao2025_raw.md`
  - `debug_madrid.md`
  - `debug_rtrvr_madrid.txt`

**Scraped raw text dumps:**
  - `_te_ao_wta_2025.md`
  - `_te_wta_2025.md`
  - `firecrawl.txt`
  - `perplexity.txt`

**Superseded / reference prompts:**
  - `codex_data_backfill_prompt.md`
  - `codex_ufc_project_prompt.md`
  - `antigravity_tennis_bet_prompt.md`
  - `gemini_data_prompt.md`

**Old secrets file (already in .gitignore):**
  - `GMAIL_secrets.txt`

**CatBoost training cache:**
  - `catboost_info/` (entire directory)

**Files to KEEP in root:**
  - `app.py`, `config.py`, `requirements.txt`, `README.md` — core project files
  - `codex_todo.md`, `claude_todo.md` — active task trackers
  - `codex_retrain_prompt.md`, `deepseek_wta_backfill_prompt.md` — active agent prompts
  - `PROJECT_BUILD_SPEC.md` — project specification
  - `rtvt.ai.txt`, `openrouter_api.txt`, `api.txt` — active API keys (used by scripts)

- [x] Add `archive/` to `.gitignore`

---

## Phase 9: Architecture Upgrades (1-3 days each)

### 9.1 SQLite canonical DB (from UFC project)
- [x] **Read-side done**: `src/sqlite_storage.py` provides `load_matches_frame()`, `load_odds_frame()`, `load_prediction_log_frame()`, `load_bankroll_state_payload()`, `sync_reference_players()`, `sync_player_aliases()` — all with `fallback_to_csv=True`
- [x] **Write-side dual-write done** — pipeline/matches, ELO ratings, features, odds, predictions, and bankroll all sync into SQLite while keeping CSV/JSON artifacts as caches
- Entities already in SQLite schema: players, player_aliases, matches, odds, predictions, bankroll_log
- Still artifact-only on disk by design: model reports, calibration exports, feature importance, test predictions
- Files: `src/sqlite_storage.py`, `src/data_pipeline.py`, `src/elo_engine.py`, `src/feature_engineering.py`, `src/odds_scraper.py`

### 9.2 Multi-bookmaker odds aggregation
- [x] Support multiple odds sources with fair probability computed as median across bookmakers
- UFC uses median of 2+ bookmakers + min bookmaker count filter
- Currently tennis takes whatever single odds source is available
- Would materially improve edge estimation reliability
- Files: `src/odds_scraper.py`, `src/value_engine.py`, `src/sqlite_storage.py`, `app.py`

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

## Execution Order for Remaining Tasks

### 🔴 Batch A — DO NOW (data quality chain)
1. **12.2** Fix rtrvr.ai 401
2. **12.3** Re-run WTA backfill + retrain
3. **11.4** WTA tour-specific ensemble weights (right after retrain)

### 🟡 Batch B — NEXT (email improvements, all independent)
4. **11.1** Model info + accuracy in daily email
5. **11.2** Verify weekly retrain + add skip log
6. **11.3** 30-day strategy forecast in email

### 🟢 Batch C — LATER (advanced / hardening)
7. **12.1-final** Flip `fallback_to_csv=False` (after stability proven)
8. **10.1** Prediction intervals / uncertainty
9. **10.2** Serve/return feature expansion
10. **10.3** In-play odds movement tracking

---

## Phase 10: Advanced (1+ week each)

### 10.1 Prediction intervals / uncertainty quantification
- [x] Output prediction intervals instead of just point estimates via a saved split-conformal residual artifact per tour; predictor now emits lower/upper bounds and interval width alongside `ensemble_prob_p1`
- Options: conformal prediction, bootstrap ensembles, or Bayesian approaches
- Helps filter bets where the model is uncertain even if the point estimate looks good
- Files: `src/predictor.py`, `src/model_training.py`

### 10.2 Feature expansion: serve/return stats
- [x] Expand serve/return features from existing match stats with rolling and surface-specific first-serve win%, second-serve win%, service-points won%, return-points won%, ace%, and BP-save%
- Current serve features (ace%, 1st serve%, BP save%) are rolling-20 averages
- Surface-specific serve metrics would add signal (especially for grass vs clay)
- Files: `src/feature_engineering.py`

### 10.3 In-play odds movement tracking
- [x] Track odds movement between scraping windows via `src/odds_tracker.py`; `refresh_odds()` now exports `data/odds/odds_movement.csv` with prior-vs-latest deltas per match
- Steam moves (sharp money) are a strong signal the model can incorporate
- Requires more frequent odds snapshots (e.g. every 2 hours on match day)
- Files: `src/odds_scraper.py`, new `src/odds_tracker.py`

### 10.4 Automated performance dashboard
- [x] Weekly email/HTML summary of model performance: actual ROI, calibration drift, hit rate by confidence tier
- Catch model degradation early instead of discovering it after losses
- Files: `src/daily_report.py` or new `src/performance_report.py`

---

## Phase 11: Email & Reporting Improvements

### 11.1 Add model info + prediction accuracy to daily email
- [x] Add a compact "Model Info" section to the HTML email report in `_build_html_report()`, sourced from `models/model_report_{tour}.json`
- Include for each tour (ATP/WTA):
  - Model type: "CatBoost/XGBoost ensemble (60/40 blend)"
  - Test set accuracy: e.g. "64.6% accuracy on 2,982 test matches"
  - Log-loss and ECE (calibration error)
  - Last trained date
  - Training data window (e.g. "trained on 189,745 matches, tested 2025-01-06 to 2026-03-13")
- Read these values from `models/model_report_{tour}.json` (already generated by `train_models()`)
- Place this section AFTER the recommendations table and BEFORE warnings
- Keep it compact: a small table or 2-3 lines per tour
- Files: `src/daily_report.py` (in `_build_html_report()` around line 460)

### 11.2 Verify retrain frequency is weekly (not daily)
- [x] Confirmed the current retrain policy in `maybe_retrain_models()` only triggers weekly (RETRAIN_WEEKLY_DAYS=7) or on 50+ new matches — NOT every daily run; added a skip log with days-since-train and new-row counts
- The current implementation already does this correctly via `training_state.json` timestamps
- Add a one-line log message when retrain is SKIPPED: `log.info("Retrain skipped: last trained {days_ago}d ago, {new_rows} new matches (threshold: {RETRAIN_MATCH_THRESHOLD})")`
- This gives visibility in GitHub Actions logs that retraining was evaluated but skipped
- Files: `src/model_training.py` (in `maybe_retrain_models()`)

### 11.3 Add 30-day betting strategy forecast to email
- [x] Add a "30-Day Strategy Outlook" section to the daily email with conservative / expected / optimistic projections and an insufficient-history fallback
- Calculate expected capital multiplier over 30 betting days based on:
  - Historical hit rate from `prediction_log.csv` (last 90 days of actual results)
  - Average edge per bet (from historical bets)
  - Average number of bets per day
  - Kelly fraction used (0.25)
- Formula approach:
  - `daily_expected_return = avg_bets_per_day * avg_kelly_fraction * avg_edge`
  - `30_day_multiplier = (1 + daily_expected_return) ^ 30`
  - Show optimistic (using upper confidence bound) and conservative (using lower bound) scenarios
- If insufficient historical data (<30 bet results), show: "Insufficient betting history for 30-day forecast. Need 30+ resolved bets."
- Example output in email:
  ```
  30-Day Strategy Outlook (based on last 90 days):
  - Resolved bets: 47 | Hit rate: 62.3% | Avg edge: 8.1%
  - Conservative estimate: EUR 100 → EUR 118 (×1.18)
  - Expected estimate:     EUR 100 → EUR 134 (×1.34)
  - Optimistic estimate:   EUR 100 → EUR 152 (×1.52)
  ```
- Files: `src/daily_report.py` (new function `_compute_30day_forecast()`), read from `prediction_log.csv`

### 11.4 WTA ensemble weight: consider tour-specific blending
- [x] The predictor now supports tour-specific CatBoost/XGBoost fallback weights: ATP stays 0.6/0.4, WTA uses 0.3/0.7 when the active ensemble is the standard CatBoost/XGBoost blend
- Current: global `CATBOOST_WEIGHT=0.6` / `XGBOOST_WEIGHT=0.4` in config.py used for both tours
- Implement tour-specific weights: add `CATBOOST_WEIGHT_WTA` and `XGBOOST_WEIGHT_WTA` to config.py
- Default to current values for ATP, use 0.3/0.7 for WTA (best from latest sweep)
- Update `src/predictor.py` to check for tour-specific weight config
- Files: `config.py`, `src/predictor.py`

---

## Phase 12: Remaining Infrastructure

### 12.1 Complete SQLite migration — write paths + drop CSV source-of-truth
- [x] Dual-write is in place for canonical data flows: matches, ELO, features, odds, predictions, and bankroll now sync to SQLite alongside CSV/JSON caches.
- [x] Final cleanup: core app/report/training/prediction consumers now read with `fallback_to_csv=False`; CSV caches remain on disk for portability, exports, and rare ingest/debug workflows.

**Completed dual-write paths:**
| Module | CSV written | Lines | What to do |
|--------|------------|-------|------------|
| `src/elo_engine.py` | `{tour}_elo_ratings.csv`, `{tour}_elo_snapshot.csv` | 251, 274 | Done — `sync_elo_ratings()` writes ratings + snapshot into SQLite while keeping CSV cache |
| `src/feature_engineering.py` | `{tour}_player_features.csv` | 622, 703 | Done — `sync_features_frame()` writes features into SQLite while keeping CSV cache |
| `src/data_pipeline.py` | `{tour}_matches_master.csv` | 335 | Done — pipeline already calls `sync_matches_frame()` after writing the CSV |
| `src/model_training.py` | calibration, feature_importance, test_predictions CSVs | 1632-1674 | Keep as CSV — these are analysis artifacts, not canonical data flow inputs |

**Medium priority — verify/monitor but already dual-writing:**
| Module | CSV written | SQLite equivalent | What to do |
|--------|------------|-------------------|------------|
| `src/odds_scraper.py` | ODDS_UPCOMING, ODDS_HISTORY | `sync_odds_frame()` exists | Done — odds scraper now syncs both history and current snapshots right after CSV write |
| `src/daily_report.py` | PREDICTION_LOG_FILE | `sync_prediction_log_frame()` exists | Already syncs — keep the startup-before-read ordering |

**Low priority — backfill/ingest tools (run rarely):**
| Module | CSV written | What to do |
|--------|------------|------------|
| `src/data_updater.py` | Raw match CSVs, player files | Keep CSV — these are ingestion landing files, not query targets |
| `src/wta_backfill.py` | Raw WTA CSVs | Keep CSV — same as above |
| `src/tml_ingest.py` | Raw ATP CSVs | Keep CSV — same as above |

**Migration pattern:** dual-write is now in place; the remaining optional hardening step is flipping selected consumers to `fallback_to_csv=False` once SQLite stability is proven.

Files: `src/sqlite_storage.py`, `src/elo_engine.py`, `src/feature_engineering.py`, `src/data_pipeline.py`, `src/odds_scraper.py`

### 12.2 Fix rtrvr.ai 401 — investigate API key
- [x] The rtrvr.ai fallback scraper was verified live: current key/auth returns HTTP 200 and Tennis Explorer pages return accessibility-tree payloads; `wta_backfill.py` now logs 401/tab-level failures clearly and skips gracefully
- API key in `rtvt.ai.txt`: `rtrvr_aGzcuAjQoySym4UtktIbCro1LgXgh24U2218MJEYzp4`
- Possible causes: key expired, wrong auth header format, account issue
- Test with: `curl -X POST https://api.rtrvr.ai/scrape -H "Authorization: Bearer <key>" -H "Content-Type: application/json" -d '{"urls":["https://www.google.com"]}'`
- If key is dead: log a clear warning and skip rtrvr.ai gracefully (don't crash)
- Files: `src/wta_backfill.py` (`rtrvr_scrape()` and `_get_rtrvr_key()`)

### 12.3 Re-run WTA backfill for missing 9 tournaments
- [x] Re-ran the backfill successfully on 2026-03-19: `wta_matches_2025.csv` now has 1,853 rows and `wta_matches_2026.csv` has 491 rows; duplicates=0 and `R128` in small draws=0
  ```bash
  del data\raw\tennis_wta\wta_matches_2025.csv
  del data\raw\tennis_wta\wta_matches_2026.csv
  python -m src.wta_backfill --years 2025 2026
  ```
- Expected: ~1,800-2,200 matches for 2025 (currently 1,337), ~480-550 for 2026
- Missing tournaments: Madrid, Rome, Canada, Cincinnati, Abu Dhabi, Seoul, Singapore, Austin, Bad Homburg
- After backfill: retrained ATP/WTA artifacts on 2026-03-19 and ran `run_backtest(tours=('atp', 'wta'))`; both tours hit the expected in-sample guard with the new train cutoff instead of failing
- Files: `src/wta_backfill.py`, `data/raw/tennis_wta/`

---

## Phase 13: CI Fix — GitHub Action Timeout (2026-03-19)

**Problem:** `daily_bet.yml` runs for ~2 hours then fails. The `logreg` and `elasticnet` models train on ~190K unscaled feature rows using lbfgs/saga solvers, which fail to converge within `max_iter` and loop repeatedly across 6 models × 3 CV folds × 2 tours = 36 fits. Additionally, sklearn 1.8 deprecation warnings spam the output.

### 13.1 Wrap logreg/elasticnet in StandardScaler pipeline
- [x] In `src/model_training.py`, add imports: `from sklearn.pipeline import Pipeline` and `from sklearn.preprocessing import StandardScaler`
- [x] In `_fit_model()` (line ~925), replace the separate `elasticnet` and `logreg` branches with a single branch:
  ```python
  if model_name in ("elasticnet", "logreg"):
      model = Pipeline([
          ("scaler", StandardScaler()),
          ("clf", LogisticRegression(**params)),
      ])
      model.fit(matrices["x_train_tab"], matrices["y_train"])
      return model
  ```
- **Why:** Features include raw rankings (1-500+), ELO ratings (1000-2500+), and binary dummies. Without scaling, lbfgs/saga cannot converge and grinds for hours.
- [x] Verified locally: `_predict_model_proba()` still works with Pipeline objects, pickling/unpickling is stable, and `_feature_importance_rows()` now unwraps the final pipeline estimator so logistic-model importance exports still populate.

### 13.2 Fix sklearn penalty deprecation
- [x] In `_default_model_params()` (line ~769), for `elasticnet`: remove `"penalty": "elasticnet"` — sklearn 1.8+ uses `l1_ratio` to control this. Keep `l1_ratio: 0.5` and `solver: "saga"`.
- [x] For `logreg` (line ~780): add `"l1_ratio": 0` to make L2 penalty explicit without using the deprecated `penalty` param.
- [x] Bump `max_iter` to `3000` for both (from 2000/1000) as a safety margin with scaling.
- **Why:** `penalty` param deprecated in sklearn 1.8, will be removed in 1.10. Current warnings appear 8+ times per CI run.

### 13.3 Reduce Sackmann 404 log noise
- [x] In `src/data_updater.py` line ~176 (`_fetch_from_fallback_source`): change `logger.info(` to `logger.debug(`
- [x] Line ~227 (`http_fallback_update`): change `logger.warning(str(exc))` to `logger.debug(str(exc))` for `SackmannYearUnavailableError`
- [x] Line ~311 (`download_years_http_fallback`): same change — `logger.warning` → `logger.debug` for `SackmannYearUnavailableError`
- **Why:** 2025/2026 Sackmann CSV files return HTTP 404 because they're not published yet. The code handles this correctly (returns empty frame), but logs 4 WARNING lines per run that clutter CI output.

### 13.4 Verify CI after fixes
- [ ] Push changes and re-run `daily_bet.yml` workflow
- [ ] Confirm: (a) completes in <10 min, (b) no ConvergenceWarning, (c) no FutureWarning about penalty, (d) email sends successfully
- [ ] Compare model metrics before/after to ensure StandardScaler didn't degrade accuracy
- Local verification done on 2026-03-19: targeted WTA fits for `elasticnet` and `logreg` on recent feature rows completed with zero `FutureWarning` and zero `ConvergenceWarning`; Pipeline `predict_proba()` and pickle round-trip also passed.

---

## Bug Tracker

### Known bugs
- [x] `generate_recommendations()` default capital is `5.0` (line 205 of value_engine.py) -- should be `DEFAULT_CAPITAL`
- [x] ~~Streamlit app double-logging~~ -- verified: `_log_recommendations_to_prediction_log()` in app.py has deduplication via `existing_ids` set
- [x] `bankroll_log.json` has UTF-8 BOM encoding issue (fixed: BOM stripped from file; tolerant readers use `utf-8-sig`)
- [x] Custom match entry in app.py accepts arbitrary player IDs without validation against known players

### Resolved bugs
- [x] Staleness calculation used only raw Sackmann dates, ignoring processed/custom matches (fixed: now takes max of all sources)
- [x] Capital hardcoded at EUR 5.00 (fixed: updated to EUR 100.00 in config + bankroll_log)
- [x] `bankroll_log.json` UTF-8 BOM caused JSON decode failures in non-tolerant readers
