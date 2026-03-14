# CLAUDE TODO - TennisBet QA & Strategic Review

## Phase 1 Review: Data Foundation
- [ ] Review `data_updater.py`: git pull -> HTTP fallback logic, timeout handling, error messages
- [ ] Verify staleness thresholds: are 3/7/14 day cutoffs reasonable for tennis calendar?
- [ ] Review custom match entry: does it match Sackmann schema exactly? Is `source="custom"` preserved through pipeline?
- [ ] Test HTTP fallback: manually break git path, confirm CSV download works
- [ ] Review `data_pipeline.py`: schema correctness, missing value handling, retirement flagging
- [ ] Verify custom matches merge correctly into master (no duplicates, proper ordering)
- [ ] Review `elo_engine.py`: verify chronological computation, no future data leakage
- [ ] Verify incremental ELO parity: resume from last state vs full recompute
- [ ] Spot-check ELO outputs: top-ranked players should have strongest ELOs; surface ELOs diverge logically
- [ ] Validate processed CSVs: row counts, date ranges, no duplicates

## Phase 2 Review: Features & Models
- [ ] Review `feature_engineering.py`: confirm all features use only pre-match data
- [ ] Verify streak features reset correctly across tournaments/seasons
- [ ] Verify home country mapping with spot checks
- [ ] Check neutral handling for low-sample home features
- [ ] Leakage audit: rolling windows do not include current match
- [ ] Review `model_training.py`: strict temporal split, no shuffle leakage
- [ ] Validate Optuna outcomes and hyperparameter sanity
- [ ] Review feature importances for suspicious leakage proxies
- [ ] Check calibration: predicted probabilities vs empirical outcomes

## Phase 3 Review: Prediction & Value
- [ ] Review `predictor.py`: ensemble weights and merge logic correctness
- [ ] Review `odds_scraper.py`: player name matching robustness and rate limiting
- [ ] Review `value_engine.py`: overround removal and allocation caps
- [ ] Review `backtest.py`: look-ahead bias prevention and bankroll math
- [ ] Validate backtest ROI realism

## Phase 4 Review: App
- [ ] Test Streamlit app end-to-end
- [ ] Test "Update Data" flow and staleness banner transitions
- [ ] Test custom match form with fake data and downstream visibility
- [ ] Test forced stale metadata (>14 days) and critical warning rendering
- [ ] Verify `bankroll_log.json` persists across app restarts
- [ ] Verify `prediction_log.csv` records every prediction with timestamps/outcomes
- [ ] Verify "SKIP TODAY" logic when no value bets exist
- [ ] Verify allocation sums/caps respect configured limits
- [ ] Check edge cases: only ATP/WTA, one match only, no matches
- [ ] UX review: recommendation cards are clear for quick decisions

## Strategic Checks
- [ ] Compare CatBoost vs XGBoost disagreement rate; if >30%, review ensemble weighting
- [ ] If backtest ROI < 0%, flag for model revision before live usage
- [ ] If backtest ROI > 20%, flag overfit risk and investigate
- [ ] Verify WTA robustness vs ATP; may require higher edge threshold

## Phase 6 Review: GitHub Actions + Email
- [ ] Review `daily_report.py`: full headless run without Streamlit dependency
- [ ] Review email HTML template for mobile readability and key info completeness
- [ ] Verify `.gitignore`: no secrets, no raw data, no large model binaries
- [ ] Review GitHub Action workflow: cron timing and Chrome dependencies
- [ ] Security audit: all secrets provided via GitHub Secrets, nothing hardcoded
- [ ] Test CI failure mode: scraping failures still send diagnostic email
- [ ] Verify `prediction_log.csv` artifact upload for auditability
