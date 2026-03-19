# TennisBet Project Build Spec

## Purpose

TennisBet is a local-first tennis prediction and value-betting system. It was built to:

- ingest ATP/WTA historical match data
- backfill recent matches when the main historical source lags
- compute pre-match features without leakage
- train match-outcome models
- join predictions with bookmaker odds
- identify positive-edge bets with capped bankroll allocation
- run as both an interactive app and a headless daily report job

This file is the closest thing to a full implementation spec for the current project.

## High-Level Architecture

Core flow:

1. `src/data_updater.py`
   Pulls raw ATP/WTA data from local Sackmann clones, falls back to HTTP downloads, and tracks freshness metadata.
2. `src/data_pipeline.py`
   Builds clean master match files by merging raw yearly files and custom/backfilled matches.
3. `src/elo_engine.py`
   Computes chronological ELO and surface ELO ratings.
4. `src/feature_engineering.py`
   Builds supervised learning rows using only pre-match information.
5. `src/model_training.py`
   Trains CatBoost + XGBoost models on a temporal split.
6. `src/predictor.py`
   Scores either historical feature files or upcoming matches from current odds.
7. `src/value_engine.py`
   Removes overround, computes betting edge, and allocates bankroll.
8. `src/backtest.py`
   Evaluates betting strategies on held-out predictions with historical odds.
9. `app.py`
   Streamlit UI for manual operation.
10. `src/daily_report.py`
    Headless daily cycle for CI/email.

## Data Layout

- `data/raw/tennis_atp/`, `data/raw/tennis_wta/`
  Historical yearly source CSVs.
- `data/custom/`
  Custom/backfilled match CSVs appended into the master set.
- `data/processed/`
  Master match tables, ELO tables, snapshots, feature tables, and generated predictions.
- `data/odds/`
  Upcoming odds and historical odds.
- `data/meta/`
  Last-update metadata, bankroll log, prediction log, leakage audits.
- `models/`
  Model binaries, preprocess payloads, calibration files, feature importance, test predictions, reports, backtests.

## Data Sources

Primary:

- Jeff Sackmann ATP/WTA match repositories cloned locally under `data/raw/`

Secondary / recency backfill:

- Flashscore finished-match scraping for recent results
- TML-based ATP ingestion / Tennis Explorer WTA backfill work added later in the project

Odds source:

- Flashscore upcoming odds scraping
- manual odds CSV upload fallback

## Match Master Construction

Implemented in `src/data_pipeline.py`.

Key rules:

- rows missing `winner_id`, `loser_id`, or `score` are dropped
- retirements and walkovers are preserved in the data, but flagged as not training-eligible
- `match_date` is derived from `tourney_date`
- `match_key` is a stable dedup identity built from:
  - `tourney_id`
  - `tourney_date`
  - `winner_id`
  - `loser_id`
  - `score`
- full rebuilds and incremental runs both deduplicate on `match_key`

Reasoning:

- earlier dedup keys based on `match_num` / `round` were not stable enough across backfills and custom imports
- using a scrape-stable identity avoids duplicate fights/matches surviving re-runs

## ELO Design

Implemented in `src/elo_engine.py`.

Settings:

- `ELO_START = 1500`
- `ELO_K_BASE = 32`
- blended expectation uses:
  - `0.7 * overall_elo`
  - `0.3 * surface_elo`
- K-factor adjusts for:
  - tournament level
  - round
  - best-of-five matches

Why ELO is used:

- it is a compact, leakage-safe skill estimate
- it handles sparse players better than raw rolling stats alone
- it gives a stronger signal than rankings when rankings are missing or stale

## Feature Set

Implemented in `src/feature_engineering.py`.

The project builds pre-match features in these groups:

1. Player strength
   - `p1_rank`, `p2_rank`
   - overall ELO
   - surface ELO
   - ELO differentials

2. Recent form
   - rolling win % over 5 / 10 / 20 matches
   - surface-specific rolling win %
   - current win / loss streaks
   - 5-match streak summary

3. Head-to-head
   - total H2H
   - H2H wins by player
   - H2H win %
   - surface-specific H2H wins

4. Tournament / experience
   - tournament wins so far in the current event
   - matches played before the current match
   - titles in the last 12 months

5. Home / location
   - player home-country flag
   - directional home advantage flag
   - home-court historical win %

6. Rest / workload
   - days since last match
   - matches in last 14 days
   - sets played in last 7 days

7. Serve profile
   - ace rate
   - first serve in %
   - break-point save %

8. Context
   - surface
   - tournament level
   - round
   - best-of format

Leakage controls:

- features are generated before any state update from the current match
- H2H state is read before the current match is added
- rolling windows exclude the current match
- `p1/p2` ordering was explicitly changed to be outcome-independent when rankings are missing

Why these statistics were chosen:

- rankings and ELO capture broad player quality
- recent-form windows capture short-term momentum
- H2H captures matchup-specific effects
- workload/rest captures fatigue
- serve metrics are high-signal tennis-specific micro-stats
- surface and tournament context matter materially in tennis

Default fallback values:

- win % defaults to `0.5`
- home win % defaults to `0.5`
- ace % defaults to `0.06`
- first-serve % defaults to `0.62`
- break-point save % defaults to `0.60`
- days-since-last defaults to `14`

## Model Training

Implemented in `src/model_training.py`.

Models:

- CatBoostClassifier
- XGBClassifier

Split:

- strict temporal split at `TRAIN_CUTOFF = "2025-01-01"`
- separate train-core / validation fold inside the training window
- test set is evaluation-only

Ensemble:

- current config defaults:
  - `CATBOOST_WEIGHT = 0.6`
  - `XGBOOST_WEIGHT = 0.4`

Metrics tracked:

- accuracy
- log-loss
- Brier score
- expected calibration error

Artifacts saved:

- model binaries
- preprocess JSON
- calibration CSV
- feature-importance CSV
- held-out test predictions
- per-tour model report JSON
- cumulative `models/model_metrics.csv`

## Prediction Logic

Implemented in `src/predictor.py`.

Two modes:

- historical prediction from feature CSV
- upcoming prediction from current odds + latest player states

Output columns:

- `ensemble_prob_p1`
- `catboost_prob`
- `xgboost_prob`
- `confidence_tier`
- `model_agreement`

## Value Betting Logic

Implemented in `src/value_engine.py`.

Process:

1. load odds + predictions
2. normalize bookmaker implied probabilities by removing overround
3. compute edge for both sides
4. choose the higher-edge side
5. filter by `MIN_EDGE_THRESHOLD`
6. allocate stake with caps

Key controls:

- `MIN_EDGE_THRESHOLD = 0.05`
- `MAX_DAILY_BETS = 3`
- `MAX_DAILY_CAPITAL_PCT = 0.50`
- `MIN_BET_AMOUNT = 0.50`

Allocation logic:

- base stake % starts from 10%
- stake increases with edge
- confidence tier multiplies allocation:
  - HIGH = `1.3x`
  - MEDIUM = `1.0x`
  - LOW = `0.7x`
- clipped by number of bets and overall daily capital cap

## Backtesting

Implemented in `src/backtest.py`.

Strategies:

- dynamic value-based staking
- flat favorite
- flat random baseline

Current limitation:

- ROI validation depends on overlap between held-out predictions and `data/odds/odds_history.csv`
- if there is no overlap, backtest result is `no_odds_overlap`

## UI and Automation

Interactive:

- `app.py` Streamlit app

Headless:

- `src/daily_report.py`
- GitHub Actions workflow for scheduled runs

Daily report includes:

- generated timestamp
- bankroll
- data freshness
- last new match
- analyzed matches
- value bets or skip message
- warnings / errors

## Freshness / Staleness

Freshness metadata lives in `data/meta/last_update.json`.

Important nuance:

- raw-source freshness can differ from processed-data freshness when recent matches were backfilled outside the Sackmann repos
- the project was patched so `staleness` is recomputed from `last_new_match`, not just cached from older raw-source checks

## Todo Management

There is no formal issue tracker inside the repo. Work was managed through:

- `codex_todo.md`
  Build-phase checklist and implementation progress.
- `claude_todo.md`
  Review findings, QA notes, and strategic concerns.
- prompt files such as:
  - `codex_data_backfill_prompt.md`
  - `codex_retrain_prompt.md`
  - `deepseek_wta_backfill_prompt.md`

The practical workflow was:

1. define a phase or corrective task in markdown
2. execute it in code
3. validate with generated artifacts and reports
4. update todo/review files manually

## Secrets / API Key Handling

Secrets are intentionally not stored in git.

Ignored local files:

- `GMAIL_secrets.txt`
- `gmail_secrets.txt`
- `openrouter_api.txt`
- `api.txt`
- `.streamlit/secrets.toml`
- `.env`

Documented environment variables:

- `EMAIL_TO`
- `EMAIL_FROM`
- `EMAIL_PASSWORD`
- `SMTP_HOST`
- `SMTP_PORT`

Patterns used:

- local fallback loading from `GMAIL_secrets.txt`
- `.env.example` to document required env vars
- GitHub Actions uses GitHub Secrets instead of checked-in credentials

## Current Known Gaps

- Optuna tuning is still optional/deferred
- odds-history coverage is still limited for robust ROI validation
- broader logging/error-handling standardization is incomplete
- some repo documentation is still fragmented across todo/review/prompt files

## Minimal Rebuild Order

If rebuilding this project from scratch, the minimal order is:

1. set up directories and `config.py`
2. add raw data updater
3. build master pipeline with dedup
4. compute ELO
5. build feature engineering with leakage controls
6. train temporal models
7. implement prediction + value logic
8. add backtest
9. add automation / reporting
10. add UI last
