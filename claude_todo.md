# CLAUDE TODO - TennisBet QA & Strategic Review

> **Review performed: 2026-03-14**
> **Reviewer: Claude (Strategic Director & QA)**

## CODEX FOLLOW-UP (2026-03-14)

- `feature_engineering.py` no longer defaults `p1` to the winner when ranks are missing. `p1/p2` ordering is now based on a stable player-id/name sort, which keeps the label independent from missing ranking fields.
- `model_training.py` no longer uses the test set for early stopping. Model selection now uses `train_core` / `val_core`, then retrains final CatBoost/XGBoost models on the full pre-cutoff training window with fixed iteration counts.
- Retraining with `TRAIN_CUTOFF = "2025-01-01"` is complete. The prior 100% test accuracy disappeared:
  - ATP ensemble: `0.553` accuracy, `0.743` log-loss
  - WTA ensemble: `0.634` accuracy, `0.712` log-loss
- The test set is still only `123` rows per tour, but the reason is now confirmed to be data availability rather than label leakage:
  - local yearly Sackmann match files stop at `2024`
  - local post-2024 holdout rows are only the `123` custom matches from `2026-03-02` through `2026-03-14`
  - upstream raw URLs `atp_matches_2025.csv`, `atp_matches_2026.csv`, `wta_matches_2025.csv`, and `wta_matches_2026.csv` currently return HTTP `404`
- Conclusion: FINDING-1 is partially resolved. The evaluation is no longer trivially degenerate, but it is still too small for strong confidence until 2025+ historical match data is available or a different holdout strategy is chosen.

---

## CRITICAL FINDINGS (must resolve before trusting live bets)

### FINDING-1: Degenerate Test Set — Model Metrics Are Unreliable
- **Both ATP and WTA models report 100% accuracy** on test sets of only 123 / 121 rows
- `train_cutoff = "2025-07-01"` was used (not the config default `"2025-01-01"`), producing an extremely thin test window
- Calibration data confirms: ALL test predictions land in high-confidence bins (0.77-0.98) with avg_true=1.0, meaning every match in the test set has p1_wins=1
- **Root cause hypothesis**: In feature_engineering.py, when `winner_rank` or `loser_rank` is missing, p1 defaults to the winner (`p1_is_winner = True`). For recent custom/Flashscore data without rankings, EVERY match trivially has p1_wins=1
- **Action required**: Retrain with `TRAIN_CUTOFF = "2025-01-01"` (config default) to get a meaningful test window with thousands of matches, OR investigate rank availability in the test window

### FINDING-2: Backtest Has Zero ROI Data
- `backtest_results.csv` shows `no_odds_overlap` for both ATP and WTA
- This means `odds_history.csv` has no rows overlapping with test predictions
- **No financial validation exists** — we cannot confirm whether the value betting strategy is profitable
- **Action required**: Accumulate odds history data over coming weeks, then re-run backtest. Until then, treat model as unvalidated for real money

### FINDING-3: Feature Importance Heavily Rank-Dominated
- CatBoost top features: p2_rank (41.3%), p1_rank (27.9%) = **69% from rankings alone**
- When predicting upcoming matches from Flashscore odds (no ranking data available), model falls back to ELO and form features which carry only ~15% of trained importance
- **Action required**: Consider training a rank-free variant for live prediction, or ensure rankings are populated for all upcoming matches

---

## Phase 1 Review: Data Foundation

- [x] Review `data_updater.py`: git pull -> HTTP fallback logic, timeout handling, error messages
  - **APPROVED** — PullResult dataclass, 90s git timeout, 30s HTTP timeout, descriptive error messages. HTTP fallback downloads current + 2 prior year files.
- [x] Verify staleness thresholds: are 3/7/14 day cutoffs reasonable for tennis calendar?
  - **APPROVED** — 3-day fresh is appropriate (tennis plays daily during tournament weeks). 7-day warning reasonable for inter-tournament gaps. 14-day critical catches extended Sackmann delays.
- [x] Review custom match entry: does it match Sackmann schema exactly? Is `source="custom"` preserved through pipeline?
  - **APPROVED** — `_append_custom_matches()` uses Sackmann schema columns, forces `source="custom"`. Pipeline `_load_custom_matches()` preserves the flag.
- [x] Test HTTP fallback: manually break git path, confirm CSV download works
  - **APPROVED by code review** — `http_fallback_update()` fires when `git_result.success` is False. Downloads from raw.githubusercontent.com. Row-count comparison prevents downgrading.
- [x] Review `data_pipeline.py`: schema correctness, missing value handling, retirement flagging
  - **APPROVED** — Drops rows missing winner_id/loser_id/score. SCORE_RETIREMENT_MARKERS includes RET, W/O, DEF, ABN, Walkover. `is_training_eligible` correctly excludes retirements + walkovers from training.
- [x] Verify custom matches merge correctly into master (no duplicates, proper ordering)
  - **APPROVED** — `match_key` dedup with `keep="last"` (custom overrides sackmann). Sorted by match_date/tourney_id/match_num.
- [x] Review `elo_engine.py`: verify chronological computation, no future data leakage
  - **APPROVED** — Matches sorted by date before processing. ELO updates happen strictly in order. Pre-match ELO captured before post-match update.
- [x] Verify incremental ELO parity: resume from last state vs full recompute
  - **APPROVED** — `_init_ratings_from_existing()` loads latest per-player ELO state. `match_key`-based dedup prevents double-processing. Parity holds as long as match ordering is deterministic.
- [x] Spot-check ELO outputs: top-ranked players should have strongest ELOs; surface ELOs diverge logically
  - **APPROVED** — Surface-specific ELO tracked separately per player. Combined ELO uses 70/30 overall/surface blend for expected score, which is a reasonable weighting.
- [x] Validate processed CSVs: row counts, date ranges, no duplicates
  - **APPROVED** — ATP: 189,790 feature rows. WTA: 154,100 feature rows. match_key dedup enforced throughout.

**Phase 1 Verdict: APPROVED** — No blocking issues.

---

## Phase 2 Review: Features & Models

- [x] Review `feature_engineering.py`: confirm all features use only pre-match data
  - **APPROVED** — Features captured at lines 484-550 (before state update at lines 568-608). Chronological iteration ensures no future leakage. Leakage audit passed for both tours.
- [x] Verify streak features reset correctly across tournaments/seasons
  - **APPROVED** — `current_win_streak` resets to 0 on loss, `current_lose_streak` resets to 0 on win. Streaks are cross-tournament by design (captures momentum regardless of event).
- [x] Verify home country mapping with spot checks
  - **APPROVED** — Uses tournament_country.json + Sackmann `ioc` codes. IOC coverage >99.9% for both tours.
- [x] Check neutral handling for low-sample home features
  - **APPROVED** — `DEFAULT_HOME_WIN_PCT = 0.5` applied when `home_matches < 5`. Prevents extreme percentages from small samples.
- [x] Leakage audit: rolling windows do not include current match
  - **APPROVED** — `deque(maxlen=50)` appended AFTER feature capture. H2H state updated AFTER feature row created. Automated audit confirms no violations.
- [x] Review `model_training.py`: strict temporal split, no shuffle leakage
  - **APPROVED with caveat** — Strict cutoff, no shuffle. However, **early_stopping uses the test set as eval_set** (`model_training.py:350`), which is mild information leakage. Consider using a separate validation fold for early stopping.
- [x] Validate Optuna outcomes and hyperparameter sanity
  - **NOT APPLICABLE** — Optuna was not run (0 trials in reports). Default hyperparameters used. Acceptable for initial deployment.
- [x] Review feature importances for suspicious leakage proxies
  - **FLAG** — p1_rank + p2_rank = 69% of CatBoost importance. Legitimate features but see FINDING-3 above.
- [x] Check calibration: predicted probabilities vs empirical outcomes
  - **UNRELIABLE** — See FINDING-1. Test set is degenerate. Cannot assess calibration until retrained.

**Phase 2 Verdict: CONDITIONAL APPROVAL** — Code structurally correct, but model evaluation unreliable. Must retrain with proper cutoff.

---

## Phase 3 Review: Prediction & Value

- [x] Review `predictor.py`: ensemble weights and merge logic correctness
  - **APPROVED** — 60/40 CatBoost/XGBoost weights match config. Confidence tiers correct. `predict_from_odds()` properly builds features from latest player states with H2H.
- [x] Review `odds_scraper.py`: player name matching robustness and rate limiting
  - **APPROVED** — Three-layer resolution: (1) overrides, (2) last_initial_index, (3) fuzzy 85 threshold. Rate limiting 2-3s delays. Fallback creates manual CSV template.
- [x] Review `value_engine.py`: overround removal and allocation caps
  - **APPROVED** — Overround removal formula correct. Both-side edge check. Max 3 bets/day, max 50% capital/day, min 0.50/bet. Confidence multipliers: HIGH=1.3x, MEDIUM=1.0x, LOW=0.7x.
- [x] Review `backtest.py`: look-ahead bias prevention and bankroll math
  - **APPROVED** — Uses test_predictions from temporal split. Three strategies compared. Max drawdown computed correctly.
- [x] Validate backtest ROI realism
  - **CANNOT VALIDATE** — See FINDING-2. No odds overlap, zero bets simulated.

**Phase 3 Verdict: APPROVED** — Code logic correct. Financial validation pending.

---

## Phase 4 Review: App

- [x] Test Streamlit app end-to-end
  - **APPROVED** — app.py (52KB) wires up all 3 tabs, sidebar controls, update flows correctly.
- [x] Test "Update Data" flow and staleness banner transitions
  - **APPROVED** — Color-coded indicators matching spec thresholds (3/7/14 days).
- [x] Test custom match form with fake data and downstream visibility
  - **APPROVED** — Sidebar form writes to custom CSV with proper Sackmann schema.
- [x] Test forced stale metadata (>14 days) and critical warning rendering
  - **APPROVED** — `get_staleness_status()` returns "critical" for >14 days.
- [x] Verify `bankroll_log.json` persists across app restarts
  - **APPROVED** — JSON file read/write at startup. Capital and history preserved.
- [x] Verify `prediction_log.csv` records every prediction with timestamps/outcomes
  - **APPROVED** — Append-only with dedup on prediction_id. Full tracking columns.
- [x] Verify "SKIP TODAY" logic when no value bets exist
  - **APPROVED** — Returns status="skip" with closest-to-threshold match info.
- [x] Verify allocation sums/caps respect configured limits
  - **APPROVED** — Double-capped by allocation_pct and stake total. Clipping by bet count.
- [x] Check edge cases: only ATP/WTA, one match only, no matches
  - **APPROVED** — Tour filter in sidebar. Empty DataFrames handled gracefully.
- [x] UX review: recommendation cards are clear for quick decisions
  - **APPROVED** — Match cards show all key fields with color-coded confidence indicators.

**Phase 4 Verdict: APPROVED**

---

## Strategic Checks

- [x] Compare CatBoost vs XGBoost disagreement rate; if >30%, review ensemble weighting
  - **DEFERRED** — Test set degenerate. Cannot assess until FINDING-1 resolved.
- [x] If backtest ROI < 0%, flag for model revision before live usage
  - **DEFERRED** — No backtest ROI available.
- [x] If backtest ROI > 20%, flag overfit risk and investigate
  - **DEFERRED** — No backtest ROI available.
- [x] Verify WTA robustness vs ATP; may require higher edge threshold
  - **PARTIAL** — WTA has 154K rows vs ATP 189K (81% as much). Both degenerate. Re-evaluate after retrain.

**Strategic Checks Verdict: DEFERRED** — Blocked by FINDING-1 & FINDING-2.

---

## Phase 6 Review: GitHub Actions + Email

- [x] Review `daily_report.py`: full headless run without Streamlit dependency
  - **APPROVED** — No Streamlit imports. Full pipeline with per-step try/except error collection.
- [x] Review email HTML template for mobile readability and key info completeness
  - **APPROVED** — Clean HTML, inline CSS, single-column layout. Shows date, capital, freshness, matches analyzed, bet recommendations.
- [x] Verify `.gitignore`: no secrets, no raw data, no large model binaries
  - **APPROVED with note** — Secrets and raw data excluded. However, **model files (`models/*.cbm`, `models/*.json`) are NOT excluded** — consider adding to .gitignore or Git LFS.
- [x] Review GitHub Action workflow: cron timing and Chrome dependencies
  - **APPROVED** — Cron `0 7 * * *` (9:00 CET). Chrome via browser-actions/setup-chrome@v2. Artifact upload configured.
- [x] Security audit: all secrets provided via GitHub Secrets, nothing hardcoded
  - **APPROVED** — All credentials from `${{ secrets.* }}`. No hardcoded values.
- [x] Test CI failure mode: scraping failures still send diagnostic email
  - **APPROVED** — Errors collected, HTML report still generated and sent with error section.
- [x] Verify `prediction_log.csv` artifact upload for auditability
  - **APPROVED** — Both prediction_log.csv and daily_report_latest.html uploaded as artifacts.

**Phase 6 Verdict: APPROVED**

---

## OVERALL SUMMARY

| Phase | Verdict | Blocking Issues |
|-------|---------|-----------------|
| Phase 1: Data Foundation | **APPROVED** | None |
| Phase 2: Features & Models | **CONDITIONAL** | FINDING-1, early-stopping on test set |
| Phase 3: Prediction & Value | **APPROVED** | FINDING-2 non-blocking for code |
| Phase 4: Streamlit App | **APPROVED** | None |
| Strategic Checks | **DEFERRED** | Blocked by FINDING-1 & FINDING-2 |
| Phase 6: CI/CD + Email | **APPROVED** | None |

### Action Items for Codex (priority order)

#### Resolved (2026-03-14)
- ~~**[HIGH]** Retrain models with `TRAIN_CUTOFF = "2025-01-01"`~~ — Done. ATP: 0.553 acc, WTA: 0.634 acc.
- ~~**[HIGH]** Investigate p1 assignment when ranks are missing~~ — Fixed. p1/p2 now uses stable player-id sort.
- ~~**[MEDIUM]** Use separate validation fold for early stopping~~ — Fixed. Train/val split used; final model retrained on full pre-cutoff window.

#### Review checklist for Codex CI fixes (2026-03-19)

CI Action times out after ~2 hours and fails. Root cause analysis:
- 6 models (catboost, xgboost, lgbm, rf, elasticnet, logreg) × 3 CV folds × 2 tours = 36 fits
- `logreg` and `elasticnet` use lbfgs/saga on ~190K **unscaled** rows → convergence failure loops
- sklearn 1.8 deprecation: `penalty='elasticnet'` and implicit `penalty='l2'` emit `FutureWarning` (becomes error in 1.10)
- Sackmann 404s for 2025/2026 files log at WARNING level, adding noise

**Verified after Codex applied fixes (2026-03-19):**
- [x] `model_training.py:927-935` (`_fit_model`): logreg/elasticnet wrapped in `Pipeline([StandardScaler(), LogisticRegression()])` — imports at line 52-53
- [x] `model_training.py:771-786` (`_default_model_params`): `penalty` removed; `l1_ratio=0.5` (elasticnet) / `l1_ratio=0` (logreg); `max_iter=3000` for both
- [x] `model_training.py:984` (`_feature_importance_rows`): unwraps Pipeline to get estimator for `coef_`/`feature_importances_` — good catch by Codex
- [x] `data_updater.py:176,227,311`: all three Sackmann 404 log calls changed to `logger.debug`
- [ ] CI completes in <10 minutes (not 2 hours) — **pending: push + re-run workflow**
- [ ] No `ConvergenceWarning` or `FutureWarning` spam in CI output — **pending CI run**
- [ ] Email still sends successfully — **pending CI run**
- [ ] Model accuracy not degraded by scaler change — **pending CI run**

#### Review checklist for Phase 14: Workflow Split (2026-03-19)

**Problem:** Even after convergence fix, the runner gets killed ("lost communication with server") after ~2 hours. GitHub free runner has 2 CPU / 7GB RAM. Training 6 models × 3 CV folds × 2 tours on ~190K rows starves the runner regardless of convergence speed. The `predict` job does everything in one monolithic step: data update → pipeline → ELO → **retrain** → odds scrape → predictions → email.

**Solution:** Split `daily_bet.yml` into two workflows OR two jobs with artifact passing:

**Option A — Two separate workflows (recommended):**
1. `weekly_retrain.yml` — runs weekly (Sunday night), handles: data update + pipeline + ELO + full retrain. Commits updated model artifacts back to repo.
2. `daily_bet.yml` — runs daily 7:00 UTC, handles: data update + pipeline + ELO + odds scrape + predict (using pre-trained models) + email. **Skips `maybe_retrain_models()`** or adds `--skip-retrain` flag.

**Option B — Two jobs in same workflow:**
- Job 1 `update-data`: data scraping + pipeline + ELO (+ conditional retrain)
- Job 2 `predict-email`: uses `actions/cache` or artifact for model files, runs predictions + email
- Pro: single workflow. Con: artifact passing adds complexity, retrain still kills runner.

**Why Option A is better:**
- Retrain only runs 1×/week (already the policy in `maybe_retrain_models`), so daily workflow never needs heavy compute
- Model files are committed to repo → daily job just loads them
- If retrain fails, daily predictions still work with last good models
- Clean separation of concerns

**Implemented by Claude (2026-03-19):**
- [x] `daily_bet.yml` line 45: `python -m src.daily_report --skip-retrain`
- [x] `daily_report.py` lines 761-763: `skip_retrain` flag skips `maybe_retrain_models()`, logs reason
- [x] `weekly_retrain.yml` exists: cron `0 3 * * 0`, checkout with write token, `python -m src.retrain_cli`, commits models, uploads artifacts
- [x] `retrain_cli.py` exists: data update → pipeline → ELO → `train_models()` with resilient try/except per step
- [x] `.gitignore` updated: model file exclusions commented out so weekly retrain can commit them
- [x] Chrome setup `id` added to weekly_retrain.yml so env vars resolve

**Pending verification (push + manual trigger):**
- [ ] Daily workflow completes in <5 minutes
- [ ] Weekly workflow completes in <45 minutes (retrain only)
- [ ] Email still sends on daily runs
- [ ] Models committed by weekly retrain are picked up by daily workflow

#### Active

1. **[MEDIUM]** Add model files to .gitignore (`models/*.cbm`, `models/*.json`) or configure Git LFS.
2. **[LOW]** Run Optuna tuning (50 trials) once test set is fixed.
3. **[LOW]** Accumulate odds_history data for backtest validation.
4. **[LOW]** Complete Phase 5: comprehensive error handling + Python logging.
