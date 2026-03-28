# TennisBet — Full Technical Documentation

> **Generated: 2026-03-28** | **Purpose:** Complete system reference for restoration and maintenance

---

## 1. Architecture Overview

Everything runs locally on Windows. GitHub repo is backup only (no deploy, no CI).

### Daily workflow (fast, ~30 sec)
```
streamlit run app.py → click "Refresh All" → scrapes live odds → applies models → shows bets
```

### Weekly workflow (slow, ~30-60 min)
```
retrain_weekly.bat → scrapes historical data → rebuilds pipeline/ELO/features → retrains models
```

---

## 2. Entry Points

| Command | Purpose | Duration |
|---------|---------|----------|
| `streamlit run app.py` | Daily betting UI | ~30 sec per refresh |
| `retrain_weekly.bat` | Full retrain pipeline | ~30-60 min |
| `python -m src.retrain_cli` | Same as bat, Python-only | ~30-60 min |
| `python -m src.daily_report --tours atp wta` | Generate + email HTML report | ~2 min |
| `python -m src.backtest --tours atp wta` | Backtest strategy on historical data | ~5 min |

---

## 3. Data Flow

```
┌─────────────────────── DATA INGESTION ───────────────────────┐
│                                                               │
│  data_updater.py ──→ git pull / HTTP fallback ──→ Sackmann   │
│  tml_ingest.py ────→ HTTP download ─────────────→ TML-DB     │
│  wta_backfill.py ──→ Firecrawl → rtrvr.ai ─────→ Tennis Exp │
│  data_updater.py ──→ Selenium (Chrome) ─────────→ Flashscore │
│  odds_scraper.py ──→ Selenium (Chrome) ─────────→ Flashscore │
│                                                               │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────── PROCESSING ────────────────────────────┐
│                                                               │
│  data_pipeline.py ──→ merge raw + custom → master CSV + SQLite│
│  elo_engine.py ─────→ chronological ELO ratings per surface   │
│  feature_engineering.py → 50+ features per match              │
│                                                               │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────── MODEL TRAINING (weekly only) ──────────┐
│                                                               │
│  model_training.py ──→ CatBoost + XGBoost + LightGBM         │
│                     ──→ + LogReg + ElasticNet (fallback)      │
│                     ──→ Optuna hyperparameter tuning           │
│                     ──→ 3-fold temporal CV                     │
│                     ──→ ensemble weight sweep → best combo     │
│                     ──→ saves to models/                       │
│                                                               │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────── PREDICTION (daily) ────────────────────┐
│                                                               │
│  predictor.py ──────→ loads models → ensemble probabilities   │
│  value_engine.py ───→ edge = model_prob - implied_prob        │
│                     → fractional Kelly sizing                  │
│                     → filters: min edge 5%, max 3 bets/day    │
│                                                               │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
┌─────────────────────── OUTPUT ────────────────────────────────┐
│                                                               │
│  app.py (Streamlit) → betting suggestions + bankroll tracker  │
│  daily_report.py ───→ HTML email via Gmail SMTP               │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. Module Reference

### 4.1 Data Ingestion

#### `src/data_updater.py` (918 lines)
**Source:** Sackmann GitHub repos (ATP + WTA historical match data)
**Method:** git pull → HTTP fallback (requests)
**Incremental:** No — downloads last 3 years, compares row counts
**Key functions:**
- `update_data_sources(dry_run) → dict` — main entry, tries git then HTTP
- `scrape_flashscore_results() → dict` — Selenium scrape of today's finished matches
- `get_staleness_status(last_match_date, today) → dict` — fresh/warning/stale check

**Staleness thresholds:** 3 days = fresh, 7 days = warning, 14 days = stale

#### `src/tml_ingest.py` (489 lines)
**Source:** TML-Database GitHub (ATP only, supplements Sackmann)
**Method:** HTTP requests
**Incremental:** Yes — merges new rows, deduplicates by match key
**Key functions:**
- `ingest(years, check_only) → dict` — downloads, converts TML→Sackmann schema, writes CSV
- `convert_to_sackmann(df) → pd.DataFrame` — column mapping + fuzzy player ID matching

#### `src/wta_backfill.py` (1,299 lines)
**Source:** Tennis Explorer (WTA tournaments not in Sackmann)
**Method:** Firecrawl API → rtrvr.ai fallback → Perplexity enrichment
**Incremental:** Yes — skips tournaments already scraped, refreshes last 21 days only
**API keys required:** FIRECRAWL_API_KEY, RTRVR_API_KEY, PERPLEXITY_API_KEY
**Key functions:**
- `backfill_wta(years) → dict` — main entry: Sackmann check → scrape → enrich → write
- `firecrawl_scrape(url, api_key) → dict` — POST to api.firecrawl.dev/v1/scrape
- `rtrvr_scrape(url, api_key) → str` — POST to api.rtrvr.ai/scrape (Bearer auth)
- `perplexity_enrich(matches) → list` — fills missing player metadata via AI

**Tournament registry:** Hardcoded slugs for 2025/2026 in lines 81-134

#### `src/odds_scraper.py` (474 lines)
**Source:** Flashscore.com (upcoming match odds)
**Method:** Selenium (Chrome headless)
**Incremental:** No — overwrites upcoming_odds.csv each run, appends to history
**Key functions:**
- `refresh_odds() → dict` — main entry: scrape → fuzzy name match → sync SQLite + CSV
- `scrape_flashscore_upcoming_odds() → pd.DataFrame` — Selenium browser automation

**Requires:** Chrome + ChromeDriver installed locally

### 4.2 Processing

#### `src/data_pipeline.py` (394 lines)
**Key functions:**
- `run_pipeline(incremental) → dict` — merges raw + custom matches, parses scores, derives columns
- `build_master_for_tour(tour, incremental) → dict` — per-tour merge + normalize + sync SQLite

**Derived columns:** is_retirement, is_walkover, is_training_eligible, game counts from score parsing

#### `src/elo_engine.py` (319 lines)
**Algorithm:** ELO_START=1500, K_BASE=32, surface-specific ratings, tournament weight multipliers
**Key functions:**
- `run_elo(incremental) → dict` — computes ELO for both tours
- `compute_elo_for_tour(tour, incremental) → dict` — chronological ELO with K-factor adjustments

**Surface blend:** 70% overall / 30% surface-specific for expected score calculation

#### `src/feature_engineering.py` (852 lines)
**Features computed (~50+):**
- Win/loss % (rolling windows)
- Service stats: 1st serve %, service game %, BP save %, ace %
- Return stats: return games won %
- Surface-specific versions of above
- Head-to-head record
- Days since last match
- ELO ratings (current + delta)
- Tournament country home advantage
- Ranking (when available)

**Key functions:**
- `build_features(tours) → dict` — main entry, iterates matches chronologically

### 4.3 Model Training

#### `src/model_training.py` (1,941 lines)
**Models trained (6 total):**

| Model | Library | Notes |
|-------|---------|-------|
| CatBoost | catboost | gradient boosting, handles categoricals natively |
| XGBoost | xgboost | gradient boosting, JSON artifact |
| LightGBM | lightgbm | gradient boosting, text artifact |
| Ridge/SGD | sklearn | linear model with L2 penalty (SGDClassifier loss='log_loss') |
| LogisticRegression | sklearn | linear baseline |
| ElasticNet | sklearn | linear with L1+L2 penalty |

**Weights:** determined automatically by ensemble sweep after each weekly retrain. Saved in `model_report_{tour}.json` → `winner_row.weights`. No hardcoded weights.

**Training pipeline:**
1. Load features, split by TRAIN_CUTOFF (180 days rolling)
2. 3-fold temporal cross-validation (respects time ordering)
3. Optuna hyperparameter tuning (50 trials per model)
4. StandardScaler pipeline for LogReg/ElasticNet
5. **Systematic ensemble weight sweep** — generates all reasonable candidates automatically:
   - Every pair of models at 50/50, 70/30, and 30/70
   - Every triple at equal weights
   - Top-4 equal (if 4+ models available)
   - Single models also compete (no ensemble can win unfairly)
   - ~40-60 candidates tested per tour
6. Selects winner by lowest cross-validated log-loss
7. Saves winning weights to `model_report_{tour}.json` → `winner_row.weights`
8. Saves all artifacts to `models/`

**✅ FIXED (BUG-4):** Predictor now reads winning weights from `models/ensemble_config_{tour}.json` (or `model_report_{tour}.json` fallback). Hardcoded weight constants removed from `config.py`.

**Key functions:**
- `train_models(tours, use_optuna) → dict` — main entry
- `maybe_retrain_models() → dict` — conditional: weekly (7 days) or 50+ new matches

**Artifacts saved per tour:**
- `catboost_{tour}.cbm` — CatBoost binary
- `xgboost_{tour}.json` — XGBoost JSON
- `lgbm_{tour}.txt` — LightGBM text
- `ensemble_config_{tour}.json` — winning weights + CV metrics
- `preprocess_{tour}.json` — feature columns + numeric medians
- `calibration_{tour}.csv` — isotonic calibration bins
- `uncertainty_{tour}.json` — conformal prediction intervals
- `feature_importance_{tour}.csv` — per-feature scores
- `model_report_{tour}.json` — full training metadata

### 4.4 Prediction & Value

#### `src/predictor.py` (752 lines)
**Key functions:**
- `predict_from_odds(odds_df, tour) → pd.DataFrame` — main daily path: odds → features → ensemble prob
- `predict_from_feature_file(tour, input_path, output_path, limit) → pd.DataFrame` — batch predictions
- `add_prediction_columns(df, tour) → pd.DataFrame` — adds model probs + confidence tier

**Confidence tiers:** HIGH (model agreement >80%), MEDIUM (60-80%), LOW (<60%)

#### `src/value_engine.py` (475 lines)
**Edge calculation:**
- Implied probability from bookmaker decimal odds
- Overround removal (power method normalization)
- Edge = model_prob − fair_implied_prob

**Bankroll allocation (fractional Kelly):**
- `f* = (b*p - q) / b` where b=odds−1, p=model_prob, q=1−p
- Kelly fraction: 0.25 (quarter Kelly for safety)
- Constraints: max 3 bets/day, max 50% daily capital, min EUR 0.50 per bet

**Key functions:**
- `generate_recommendations(capital) → pd.DataFrame` — filters + sizes + ranks value bets

### 4.5 Storage

#### `src/sqlite_storage.py` (1,561 lines)
**Database:** `db/tennis.sqlite`
**Tables:** players, player_aliases, tournaments, matches, odds, elo_ratings, features, prediction_log, bankroll_state

Dual-write architecture: all data flows write to both SQLite and CSV. CSV caches kept for portability.

### 4.6 Reporting

#### `src/daily_report.py` (914 lines)
**Pipeline:** load bankroll → refresh data → predict → recommend → HTML email → Gmail SMTP
**Email requires:** EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO, SMTP_HOST, SMTP_PORT
**Key functions:**
- `run_daily_report(tours, refresh_predictions) → dict`
- Supports `--skip-retrain` flag for CI use

#### `src/backtest.py` (572 lines)
Walk-forward backtesting: trains on rolling 180-day window, predicts next period, simulates Kelly betting.
Tracks: equity curve, win rate, Sharpe ratio, max drawdown, ROI.

---

## 5. Configuration

### `config.py` (76 lines)

**Key constants:**
```
ELO_START = 1500.0
ELO_K_BASE = 32.0
(Ensemble weights: no longer hardcoded — read from models/ensemble_config_{tour}.json at prediction time)
OPTUNA_TRIALS = 50
TRAIN_CUTOFF = 180 days rolling
MIN_EDGE_THRESHOLD = 0.05      # 5% minimum edge
MAX_DAILY_BETS = 3
MAX_DAILY_CAPITAL_PCT = 0.50
KELLY_FRACTION = 0.25
DEFAULT_CAPITAL = 100.00        # EUR
FUZZY_MATCH_THRESHOLD = 85
```

---

## 6. Dependencies

```
streamlit==1.53.1      # Web UI
catboost==1.2.10       # Model
xgboost==3.2.0         # Model
lightgbm>=4.0.0        # Model
scikit-learn==1.8.0    # LogReg/ElasticNet + preprocessing
optuna==4.7.0          # Hyperparameter tuning
pandas==2.3.3          # Data processing
numpy==2.4.1           # Numerics
plotly==6.6.0          # Charts
selenium==4.29.0       # Flashscore scraping
requests==2.32.5       # HTTP downloads
fuzzywuzzy==0.18.0     # Name matching
rapidfuzz>=3.0.0       # Fast fuzzy matching
python-dateutil        # Date parsing
```

**System requirements:** Chrome + ChromeDriver (for Selenium-based scraping)

---

## 7. Directory Structure

```
bet/
├── app.py                          # Streamlit daily UI (1,426 lines)
├── config.py                       # All constants and paths (76 lines)
├── retrain_weekly.bat              # Weekly retrain script
├── codex_todo.md                   # Task tracker
├── README.md                       # Quick reference
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages (libgomp1)
│
├── src/
│   ├── data_updater.py             # Sackmann + Flashscore ingestion (918 lines)
│   ├── data_pipeline.py            # Raw → master CSV merge (394 lines)
│   ├── tml_ingest.py               # TML-Database ATP ingestion (489 lines)
│   ├── wta_backfill.py             # Tennis Explorer WTA scraping (1,299 lines)
│   ├── elo_engine.py               # ELO rating computation (319 lines)
│   ├── feature_engineering.py      # 50+ match features (852 lines)
│   ├── model_training.py           # 6-model ensemble training (1,941 lines)
│   ├── predictor.py                # Model loading + prediction (752 lines)
│   ├── value_engine.py             # Edge detection + Kelly sizing (475 lines)
│   ├── odds_scraper.py             # Flashscore odds via Selenium (474 lines)
│   ├── odds_tracker.py             # Odds movement tracking (150 lines)
│   ├── sqlite_storage.py           # SQLite dual-write layer (1,561 lines)
│   ├── backtest.py                 # Walk-forward backtesting (572 lines)
│   ├── daily_report.py             # Email report generation (914 lines)
│   └── retrain_cli.py              # CLI retrain entry point (61 lines)
│
├── models/                         # Trained model artifacts (~16 MB)
│   ├── catboost_{tour}.cbm
│   ├── xgboost_{tour}.json
│   ├── lgbm_{tour}.txt
│   ├── ensemble_config_{tour}.json
│   ├── preprocess_{tour}.json
│   ├── calibration_{tour}.csv
│   ├── uncertainty_{tour}.json
│   ├── feature_importance_{tour}.csv
│   ├── model_report_{tour}.json
│   └── model_metrics.csv
│
├── data/
│   ├── raw/tennis_atp/             # Sackmann ATP CSVs (1968–present)
│   ├── raw/tennis_wta/             # Sackmann WTA CSVs (1968–present)
│   ├── processed/                  # Master CSVs, ELO, features, predictions
│   ├── odds/                       # upcoming_odds.csv, odds_history.csv, odds_movement.csv
│   ├── custom/                     # User-entered match results
│   └── meta/                       # last_update.json, bankroll_log.json, aliases, etc.
│
├── db/
│   └── tennis.sqlite               # SQLite database (dual-write with CSV)
│
├── .streamlit/
│   └── secrets.toml.example        # API keys template
│
└── archive/                        # Old/unused files (gitignored)
```

---

## 8. API Keys Required

| Key | Used by | Service | Purpose |
|-----|---------|---------|---------|
| FIRECRAWL_API_KEY | wta_backfill.py | api.firecrawl.dev | Scrape Tennis Explorer |
| RTRVR_API_KEY | wta_backfill.py | api.rtrvr.ai | Fallback scraper |
| PERPLEXITY_API_KEY | wta_backfill.py | Perplexity AI | Enrich player metadata |
| EMAIL_FROM | daily_report.py | Gmail SMTP | Send reports |
| EMAIL_PASSWORD | daily_report.py | Gmail SMTP | App password |
| EMAIL_TO | daily_report.py | Gmail SMTP | Recipient |

**For daily app.py use:** No API keys needed (Selenium scrapes Flashscore directly).
**For weekly retrain:** FIRECRAWL_API_KEY needed for WTA backfill. Others optional.

---

## 9. Scraping Strategy Summary

| Module | Source | Method | Incremental? | API keys? |
|--------|--------|--------|-------------|-----------|
| data_updater | Sackmann GitHub | git pull → HTTP | No (3 years) | None |
| flashscore_results | Flashscore.com | Selenium | No (today only) | None |
| tml_ingest | TML-Database | HTTP requests | Yes (merge) | None |
| wta_backfill | Tennis Explorer | Firecrawl → rtrvr.ai | Yes (21-day window) | FIRECRAWL, RTRVR |
| odds_scraper | Flashscore.com | Selenium | No (fresh snapshot) | None |

---

## 10. How to Restore from Scratch

1. **Clone or copy the project folder**
2. **Create venv and install deps:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Install Chrome + ChromeDriver** (for Selenium scraping)
4. **Set API keys** in `.streamlit/secrets.toml` or environment variables
5. **Run full retrain:**
   ```
   retrain_weekly.bat
   ```
   This will: download all data → build pipeline → compute ELO → engineer features → train 6 models → pick best ensemble
6. **Run daily:**
   ```
   streamlit run app.py
   ```
   Click "Refresh All" to get today's betting suggestions.

---

## 11. Total Codebase Stats

| Metric | Value |
|--------|-------|
| Python source lines | ~13,300 |
| Source modules | 15 |
| SQLite tables | 8 |
| Models trained per tour | 5 base + ensemble |
| Features per match | ~50+ |
| Historical data | ATP 1968–present, WTA 1968–present |
| Training window | 180 days rolling |
| CV folds | 3 (temporal) |
| Optuna trials | 50 per model |
