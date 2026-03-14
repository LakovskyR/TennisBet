# ANTIGRAVITY MASTER PROMPT — Tennis Betting Prediction System ("TennisBet")

## PROJECT IDENTITY

**Project**: TennisBet — ELO + Ensemble ML Tennis Match Outcome Predictor with Value Betting Engine  
**Owner**: Roman Lakovskiy  
**Build Location**: `C:\Users\lakov\OneDrive\Документы\PROJECTS\bet`  
**Target**: Streamlit app for daily match-winner betting recommendations (FDJ Parions Sport, €5 test capital)  
**Data Sources**: [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) + [JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta) + Flashscore odds scraping  

---

## AGENT ROLES

| Agent | Role | Artifact |
|-------|------|----------|
| **Codex** | Primary builder. Writes all code, pipelines, models, app. | `codex_todo.md` |
| **Claude** | Strategic director & QA reviewer. Reviews Codex output, catches logic errors, validates model coherence, signs off modules. | `claude_todo.md` |

**Workflow**: Codex builds module → marks done in `codex_todo.md` → Claude reviews → approves or returns with corrections in `claude_todo.md` → iterate until sign-off.

---

## ARCHITECTURE OVERVIEW

```
bet/
├── data/
│   ├── raw/                    # Cloned Sackmann repos (git submodule or copy)
│   │   ├── tennis_atp/
│   │   └── tennis_wta/
│   ├── processed/              # Cleaned, merged, feature-engineered CSVs
│   │   ├── atp_matches_master.csv
│   │   ├── wta_matches_master.csv
│   │   ├── atp_elo_ratings.csv
│   │   ├── wta_elo_ratings.csv
│   │   └── player_features.csv
│   ├── odds/                   # Scraped odds from Flashscore
│   │   ├── upcoming_odds.csv
│   │   └── odds_history.csv
│   ├── custom/                 # Our own match entries when Sackmann is stale
│   │   ├── custom_matches_atp.csv
│   │   └── custom_matches_wta.csv
│   └── meta/                   # State tracking across sessions
│       ├── last_update.json
│       ├── bankroll_log.json
│       └── prediction_log.csv
├── models/
│   ├── catboost_atp.cbm
│   ├── catboost_wta.cbm
│   ├── xgboost_atp.json
│   ├── xgboost_wta.json
│   └── backtest_results.csv
├── src/
│   ├── __init__.py
│   ├── data_updater.py         # Module 0: Git pull / HTTP fallback / staleness manager
│   ├── data_pipeline.py        # Module 1: Data ingestion & cleaning
│   ├── elo_engine.py           # Module 2: ELO rating calculator
│   ├── feature_engineering.py  # Module 3: Feature builder
│   ├── model_training.py       # Module 4: CatBoost + XGBoost training
│   ├── predictor.py            # Module 5: Ensemble prediction engine
│   ├── odds_scraper.py         # Module 6: Flashscore odds scraper
│   ├── value_engine.py         # Module 7: Value detection + bankroll allocation
│   ├── backtest.py             # Module 8: Historical backtest
│   └── daily_report.py         # Module 9: Headless daily email report
├── .github/
│   └── workflows/
│       └── daily_bet.yml       # GitHub Action: cron 7:00 UTC daily
├── tournament_country.json     # Tournament → ISO country mapping for home advantage
├── name_overrides.json         # Sackmann ↔ Flashscore player name mapping
├── app.py                      # Streamlit application
├── config.py                   # All constants, thresholds, paths
├── requirements.txt
├── codex_todo.md
├── claude_todo.md
└── README.md
```

---

## DATA LIFECYCLE — How the Database Works

**No external database.** Everything is local CSV files inside the `bet/` folder (synced via OneDrive). No cloud DB, no push to GitHub. The system is self-contained.

### Storage Structure

```
data/
├── raw/
│   ├── tennis_atp/          ← Git submodule (JeffSackmann/tennis_atp)
│   └── tennis_wta/          ← Git submodule (JeffSackmann/tennis_wta)
├── processed/
│   ├── atp_matches_master.csv    ← Cleaned full history
│   ├── wta_matches_master.csv
│   ├── atp_elo_ratings.csv       ← ELO time series
│   ├── wta_elo_ratings.csv
│   └── player_features.csv       ← Feature matrix ready for ML
├── odds/
│   ├── upcoming_odds.csv         ← Today's scraped odds
│   └── odds_history.csv          ← Archive of all scraped odds (grows over time)
├── custom/
│   ├── custom_matches_atp.csv    ← OUR OWN data when Sackmann is stale
│   └── custom_matches_wta.csv
└── meta/
    ├── last_update.json          ← {"sackmann_pull": "2026-03-12", "last_new_match": "2026-03-10", ...}
    ├── bankroll_log.json         ← Capital tracking across sessions
    └── prediction_log.csv        ← All past predictions + actual outcomes for tracking
```

### Update Flow (triggered by Streamlit sidebar button OR on launch if data >3 days stale)

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: PULL SACKMANN DATA                             │
│                                                         │
│  Primary: git pull inside data/raw/tennis_atp/ and wta/ │
│  Fallback: if git fails, use GitHub Raw CSV API:        │
│    https://raw.githubusercontent.com/JeffSackmann/      │
│    tennis_atp/master/atp_matches_2026.csv               │
│    (download only current year file, compare row count)  │
│                                                         │
│  → Count new rows vs last known match date              │
│  → Log result in last_update.json                       │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: STALENESS CHECK                                │
│                                                         │
│  Compare last_new_match date vs today:                  │
│                                                         │
│  ≤ 3 days: ✅ "Data is fresh"                           │
│  4-7 days: 🟡 "Data is X days behind — Sackmann may    │
│              be delayed. Predictions still valid."       │
│  8-14 days: 🟠 "Data is X days stale — consider adding │
│              recent results manually via Custom Entry."  │
│  > 14 days: 🔴 "DATA CRITICALLY STALE — Sackmann has   │
│              not updated in X days. Manual entry or      │
│              alternative source strongly recommended.    │
│              Model accuracy is degrading."               │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: CUSTOM DATA ENTRY (when Sackmann is stale)    │
│                                                         │
│  Streamlit form in sidebar — "Add Recent Match":        │
│  Fields: date, tournament, surface, round, best_of,     │
│          player_1 (autocomplete from player DB),         │
│          player_2, winner, score                         │
│                                                         │
│  OR: CSV upload matching Sackmann schema                │
│                                                         │
│  → Saved to data/custom/custom_matches_atp.csv          │
│  → Merged into master during pipeline run               │
│  → Flagged as source="custom" (not "sackmann")          │
│                                                         │
│  Alternative source: Flashscore results scraper         │
│  (scrape completed match results, not just odds)        │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: INCREMENTAL PIPELINE                           │
│                                                         │
│  Only process new matches (delta since last run):       │
│  - Append to master CSV                                 │
│  - Update ELO for new matches only (don't recompute     │
│    entire history — continue from last known state)      │
│  - Rebuild features for new matches                     │
│  - Update meta/last_update.json                         │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5: RETRAIN DECISION                               │
│                                                         │
│  IF new_matches >= 50 OR days_since_last_train >= 14:   │
│    → Full retrain (CatBoost + XGBoost, both tours)      │
│    → Takes 5-10 min, show progress bar                  │
│    → Save new models, log metrics                       │
│  ELSE:                                                  │
│    → Use existing models (instant)                      │
│    → Show "Model last trained: [date] on [N] matches"   │
└───────────────┬─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6: SCRAPE ODDS + PREDICT                          │
│  (same as before — Flashscore → value filter → display) │
└─────────────────────────────────────────────────────────┘
```

### New Module: `data_updater.py` — Data Update & Staleness Manager

**Responsibilities**:
- Execute git pull with subprocess (primary) or HTTP download (fallback)
- Detect new rows by comparing max `match_date` in processed CSVs vs raw files
- Manage `last_update.json` metadata
- Merge custom matches into master pipeline
- Return staleness status and warnings for Streamlit display
- Handle Flashscore results scraping as tertiary data source (completed matches, not just odds)

**Git pull implementation**:
```python
import subprocess

def pull_sackmann(repo_path: str) -> dict:
    """Returns {"success": bool, "new_commits": int, "error": str|None}"""
    try:
        result = subprocess.run(
            ["git", "pull", "origin", "master"],
            cwd=repo_path, capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return {"success": True, "output": result.stdout}
        else:
            return {"success": False, "error": result.stderr}
    except FileNotFoundError:
        return {"success": False, "error": "git not found — falling back to HTTP"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "git pull timeout — falling back to HTTP"}
```

**HTTP fallback implementation**:
```python
import requests
import pandas as pd
from datetime import datetime

def download_current_year_csv(tour: str = "atp") -> pd.DataFrame:
    """Download only current year's match file from GitHub raw."""
    year = datetime.now().year
    url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{tour}_matches_{year}.csv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))
```

**Custom match schema** (matches Sackmann format for seamless merging):
```
tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date,
match_num, winner_id, winner_name, loser_id, loser_name, score, best_of, round,
source  ← "custom" (our addition to distinguish from sackmann data)
```

---

## MODULE SPECIFICATIONS

### Module 1: `data_pipeline.py` — Data Ingestion & Cleaning

**Input**: Raw Sackmann CSV files (`atp_matches_20*.csv`, `wta_matches_20*.csv`)  
**Output**: `atp_matches_master.csv`, `wta_matches_master.csv`

**Tasks**:
- Concatenate all yearly match files into a single master dataset per tour (ATP/WTA)
- Standardize column names across ATP and WTA schemas
- Handle missing values: drop rows missing `winner_id`, `loser_id`, `score`; impute stats with player-level medians where available
- Parse `score` field to extract: sets won/lost, total games, retirement flag (mark `is_retirement = True`)
- Add derived columns: `match_date` (proper datetime), `days_since_epoch`, `year`, `tournament_level` (G/M/A/C/D/F mapping)
- Filter out walkovers and retirements for training data (flag but exclude)
- **Update mechanism**: Check if Sackmann repo has matches newer than our latest `match_date`. If yes, download and append new rows to master CSV. Log update timestamp.
- Export clean CSVs to `data/processed/`

**Validation checks**:
- No duplicate match IDs
- All player IDs resolve to names
- Date ordering is monotonic within tournaments

---

### Module 2: `elo_engine.py` — ELO Rating Calculator

**Implements**: Standard ELO with surface-specific adjustments

**Algorithm**:
```
K = 32 (base) * tournament_weight * match_completeness
tournament_weight: Grand Slam=1.1, Masters=1.0, ATP500=0.9, ATP250=0.85, Challenger=0.7
match_completeness: full match=1.0, retirement=0.5

Expected_A = 1 / (1 + 10^((ELO_B - ELO_A) / 400))
New_ELO_A = ELO_A + K * (Result_A - Expected_A)
```

**Surface-specific ELO**: Maintain 4 separate ELO tracks per player:
- `elo_overall`
- `elo_hard`
- `elo_clay`
- `elo_grass`

**Starting ELO**: 1500 for all players. Compute from earliest available data forward chronologically.

**Output**: `atp_elo_ratings.csv` / `wta_elo_ratings.csv` with columns: `player_id, date, elo_overall, elo_hard, elo_clay, elo_grass, matches_played`

**Critical**: ELO must be computed strictly in chronological order. No data leakage — a player's ELO at match time uses ONLY prior matches.

---

### Module 3: `feature_engineering.py` — Feature Builder

**For each match, build features for BOTH players (player_1 = higher-ranked or first-listed)**:

**ELO Features** (from Module 2):
- `p1_elo_overall`, `p2_elo_overall`
- `p1_elo_surface`, `p2_elo_surface` (surface of current match)
- `elo_diff_overall` (p1 - p2)
- `elo_diff_surface`

**Form Features** (rolling windows: last 5, 10, 20 matches):
- `p1_win_pct_5`, `p1_win_pct_10`, `p1_win_pct_20`
- `p2_win_pct_5`, `p2_win_pct_10`, `p2_win_pct_20`
- `p1_win_pct_surface_10` (last 10 on same surface)
- `p2_win_pct_surface_10`

**H2H Features**:
- `h2h_p1_wins`, `h2h_p2_wins`, `h2h_total`
- `h2h_p1_win_pct`
- `h2h_surface_p1_wins`, `h2h_surface_p2_wins` (on current surface)

**Momentum / Streak Features**:
- `p1_current_win_streak`, `p2_current_win_streak` (consecutive wins entering this match)
- `p1_current_lose_streak`, `p2_current_lose_streak`
- `p1_streak_5` (+1 per win, -1 per loss over last 5 matches, range -5 to +5)
- `p2_streak_5`
- `p1_tournament_wins_current` (wins in this specific tournament so far, 0 if R1)
- `p2_tournament_wins_current`
- `p1_title_count_12m`, `p2_title_count_12m` (tournament titles won in last 12 months)

**Home Country Advantage**:
- `p1_is_home` (binary: 1 if player nationality = tournament country, 0 otherwise)
- `p2_is_home`
- `home_advantage_flag` (1 if exactly one player is home, -1 if opponent is home, 0 if neither/both)
- `p1_home_win_pct` (historical win rate when playing in home country, rolling)
- `p2_home_win_pct`
- **Mapping**: Use Sackmann `player.csv` → `ioc` (country code) + tournament country derived from `tourney_id` or manual `tournament_country.json` lookup table. Sackmann encodes country as IOC codes (FRA, USA, ESP...), tournament country must be resolved from tournament name/location.

**Fatigue / Schedule Features**:
- `p1_days_since_last_match`, `p2_days_since_last_match`
- `p1_matches_last_14d`, `p2_matches_last_14d`
- `p1_sets_played_last_7d`, `p2_sets_played_last_7d`

**Serving Stats** (career rolling avg, last 20 matches where available):
- `p1_ace_pct`, `p2_ace_pct`
- `p1_1st_serve_pct`, `p2_1st_serve_pct`
- `p1_bp_save_pct`, `p2_bp_save_pct` (break points saved)

**Contextual Features** (categorical for CatBoost, encoded for XGBoost):
- `surface` (Hard / Clay / Grass / Carpet)
- `tournament_level` (Grand Slam / Masters / 500 / 250 / Challenger)
- `round` (F / SF / QF / R16 / R32 / R64 / R128 / RR)
- `best_of` (3 or 5)

**Target**: `p1_wins` (binary: 1 if player_1 wins, 0 otherwise)

**Critical**: ALL features must be computed using only data available BEFORE the match starts. No leakage. Validate with a time-split sanity check.

---

### Module 4: `model_training.py` — Dual Model Training

**Train-Test Split**: Strictly temporal. Train on everything before 2025-01-01. Test on 2025-01-01 to present.

**Model A — CatBoost (Primary)**:
```python
CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    cat_features=['surface', 'tournament_level', 'round', 'best_of'],
    eval_metric='Logloss',
    early_stopping_rounds=50,
    verbose=100
)
```

**Model B — XGBoost (Challenger)**:
```python
XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    reg_lambda=3,
    eval_metric='logloss',
    early_stopping_rounds=50,
    use_label_encoder=False
)
```
- One-hot encode categoricals for XGBoost

**Training procedure**:
1. Train both models on same training data
2. Log feature importances for both
3. Compute on test set: accuracy, log-loss, ROI (if combined with odds), calibration curve
4. Save models to `models/` directory
5. Train separate models for ATP and WTA (4 models total)

**Hyperparameter tuning**: Optuna with 50 trials, optimizing log-loss on temporal validation fold.

---

### Module 5: `predictor.py` — Ensemble Prediction Engine

**Ensemble logic**:
```python
prob_catboost = catboost_model.predict_proba(features)[1]
prob_xgboost = xgboost_model.predict_proba(features)[1]

# Weighted ensemble: CatBoost 60%, XGBoost 40%
ensemble_prob = 0.6 * prob_catboost + 0.4 * prob_xgboost

# Confidence tiers
if abs(prob_catboost - prob_xgboost) < 0.05:
    confidence = "HIGH"   # Models agree
elif abs(prob_catboost - prob_xgboost) < 0.12:
    confidence = "MEDIUM"  # Minor disagreement
else:
    confidence = "LOW"     # Models disagree significantly
```

**Output per match**: `player_1, player_2, ensemble_prob_p1, catboost_prob, xgboost_prob, confidence_tier, model_agreement`

---

### Module 6: `odds_scraper.py` — Flashscore Odds Scraper

**Target**: Flashscore.com tennis section — upcoming matches with pre-match odds  
**Method**: Selenium or Playwright headless browser (Flashscore is JS-rendered)  
**Fallback**: If Flashscore scraping breaks, provide manual CSV input template

**Output**: `upcoming_odds.csv`
```
match_date, tournament, surface, player_1, player_2, odds_p1, odds_p2, source_url
```

**Player name matching**: Use fuzzy matching (fuzzywuzzy, threshold 85) against Sackmann player database to resolve name variants (e.g., "A. Sinner" → "Jannik Sinner", player_id 207989)

**Scheduling**: Designed to run once daily (morning) or on-demand from Streamlit

**IMPORTANT**: Include rate limiting (2-3 second delays between requests), respect robots.txt. If Flashscore blocks, fall back to manual odds input in Streamlit sidebar.

---

### Module 7: `value_engine.py` — Value Detection + Bankroll Allocation

**Implied probability from odds**:
```python
implied_prob = 1 / decimal_odds
# Remove margin (overround normalization)
total_implied = (1 / odds_p1) + (1 / odds_p2)
fair_implied_p1 = (1 / odds_p1) / total_implied
```

**Value detection**:
```python
edge = ensemble_prob - fair_implied_p1
is_value_bet = edge >= 0.05  # Minimum +5% edge threshold
```

**Bankroll allocation logic**:
```python
def allocate_bankroll(value_bets: list, capital: float) -> dict:
    """
    Dynamic allocation based on edge size and confidence.
    
    Rules:
    - If 0 value bets found: "SKIP TODAY — no value detected"
    - If 1 bet found: allocate 20-40% of capital depending on edge
    - If 2 bets found: allocate 15-30% each, max 50% total
    - If 3 bets found: allocate 10-20% each, max 50% total
    - Never recommend more than 3 bets per day
    - Never allocate more than 50% of capital in a single day
    - Minimum bet: €0.50
    
    Allocation formula per bet:
    base_pct = 0.10  # 10% base
    edge_bonus = (edge - 0.05) * 2  # Extra allocation for stronger edges
    confidence_mult = {"HIGH": 1.3, "MEDIUM": 1.0, "LOW": 0.7}
    
    allocation_pct = min(base_pct + edge_bonus, 0.40) * confidence_mult[confidence]
    stake = round(capital * allocation_pct, 2)
    """
```

**Output per recommendation**:
```
MATCH: Sinner vs Alcaraz | Clay | Roland Garros QF
MODEL: Sinner 62.3% (CatBoost: 64.1%, XGBoost: 59.8%) | Confidence: MEDIUM
ODDS: Sinner @1.72 (implied 58.1%) | Alcaraz @2.15 (implied 46.5%)  
EDGE: +4.2% on Sinner → VALUE ✓ (after margin removal: +5.1%)
STAKE: €1.50 (30% of €5.00 capital)
EXPECTED VALUE: +€0.08 per euro wagered
```

---

### Module 8: `backtest.py` — Quick Historical Backtest

**Scope**: Last 3 months of available data (approx. 2025-Q4 or latest available)

**Process**:
1. Use temporal split: train on everything before backtest window, predict within window
2. For each match in window where odds data exists:
   - Generate ensemble prediction
   - Apply value filter (+5% edge)
   - Simulate bankroll allocation with €100 starting capital
3. Track: total bets placed, win rate, ROI, max drawdown, profit curve
4. Compare vs. flat betting, vs. always-favorite, vs. random

**Output**: `models/backtest_results.csv` + summary stats printed in Streamlit

---

### Module 9: `daily_report.py` — Headless Daily Email Report (GitHub Actions)

**Purpose**: Run the full prediction cycle without Streamlit, generate HTML email, send to Roman every morning.

**Flow** (fully headless, no UI):
1. Call `data_updater.pull_sackmann()` — update raw data
2. Call `data_pipeline.run_incremental()` — process new matches
3. Call `elo_engine.update_incremental()` — update ELO ratings
4. Call `odds_scraper.scrape_upcoming()` — get today's odds from Flashscore
5. Call `predictor.predict_batch()` — ensemble predictions for all upcoming matches
6. Call `value_engine.find_value_bets()` — filter + allocate bankroll
7. Generate HTML email body
8. Send via SMTP
9. Append to `data/meta/prediction_log.csv`

**Email template (HTML)**:
```
Subject: 🎾 TennisBet Daily — [DATE] — [N bets / SKIP]

Body:
┌──────────────────────────────────────────┐
│  TENNISBET DAILY REPORT — March 15, 2026 │
│  Capital: €4.30 | Data age: 2 days       │
├──────────────────────────────────────────┤
│                                          │
│  🟢 2 VALUE BETS DETECTED               │
│  Suggested allocation: €1.80 (42%)       │
│                                          │
│  ─────────────────────────────────────── │
│  #1  Sinner vs Alcaraz                   │
│      Roland Garros QF | Clay | Best of 5 │
│      Model: Sinner 63.2% | Odds: @1.72  │
│      Edge: +5.1% | Confidence: HIGH 🟢   │
│      → Stake: €1.10                      │
│  ─────────────────────────────────────── │
│  #2  Sabalenka vs Swiatek                │
│      Madrid SF | Clay | Best of 3        │
│      Model: Sabalenka 58.7% | Odds: @1.95│
│      Edge: +7.4% | Confidence: MEDIUM 🟡 │
│      → Stake: €0.70                      │
│  ─────────────────────────────────────── │
│                                          │
│  ⚠️ Matches analyzed: 24 (ATP: 16, WTA: 8)│
│  Model: CatBoost 60% + XGBoost 40%      │
│  Last retrained: March 10, 2026          │
└──────────────────────────────────────────┘

---
This is an automated report. Not financial advice. Gamble responsibly.
```

**If no value bets**:
```
Subject: 🎾 TennisBet Daily — [DATE] — SKIP TODAY

Body:
🔴 No value bets detected across [N] analyzed matches.
Closest to threshold: [match] with +3.2% edge (need +5%).
Recommendation: Preserve capital.
```

**SMTP configuration** (via environment variables):
```python
EMAIL_TO = os.environ["EMAIL_TO"]       # roman's email
EMAIL_FROM = os.environ["EMAIL_FROM"]   # sender address
EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]  # Gmail App Password or SendGrid key
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
```

**Error handling**: If any step fails (scraper blocked, data stale, etc.), still send email with error report instead of silently failing.

---

## STREAMLIT APP SPECIFICATION (`app.py`)

### Layout

**Sidebar**:
- 🔄 "Update Data" button (triggers data_updater.py full cycle)
- Staleness banner: color-coded based on data age (✅ ≤3d / 🟡 4-7d / 🟠 8-14d / 🔴 >14d)
- Capital input (default €5.00, adjustable)
- Tour filter: ATP / WTA / Both
- Manual odds override (CSV upload fallback)
- "Refresh odds" button (triggers odds scraper only)
- "Add Recent Match" expandable form (for custom data entry when Sackmann stale)
- "Log Today's Results" expandable form (enter bet outcomes → updates bankroll_log.json)
- Model performance summary (accuracy, ROI from backtest)
- Data status line: "Last updated: [date] | Matches in DB: [N] | Model trained: [date]"

**Main Page — 3 Tabs**:

#### Tab 1: "📊 Today's Recommendations"
- Header: Date + number of analyzed matches
- **If value bets found (1-3)**:
  ```
  🟢 BETTING SIGNAL: X matches with value detected
  
  Recommended allocation: Y% of your capital (€Z.ZZ total)
  
  [Match Card 1]
  [Match Card 2]
  [Match Card 3]
  ```
  Each match card shows: players, tournament, surface, round, model probability, odds, edge %, recommended stake, confidence tier, model agreement gauge

- **If no value bets**:
  ```
  🔴 SKIP TODAY — No value bets detected across X analyzed matches.
  Closest to value: [match] with edge of +3.2% (below 5% threshold)
  Recommendation: Preserve your capital for a better opportunity.
  ```

#### Tab 2: "📈 Model Performance"
- Backtest results: profit curve chart, key metrics table
- Calibration plot (predicted prob vs actual win rate)
- Feature importance charts (CatBoost + XGBoost side by side)
- Recent prediction log (last 30 predictions with outcomes)

#### Tab 3: "⚙️ Data Status"
- Last data update timestamp
- Sackmann data freshness check
- ELO ratings last computed date
- Model last retrained date
- Odds scraper status / last run

### Design Notes
- Use `st.metric()` for key numbers
- Color-code edges: green ≥8%, yellow 5-8%, red <5%
- Match cards use `st.container()` with `st.columns()` for clean layout
- Add emoji indicators for confidence: 🟢 HIGH, 🟡 MEDIUM, 🔴 LOW

---

## CONFIG.PY — Central Constants

```python
# Paths
DATA_DIR = "data/"
RAW_ATP = "data/raw/tennis_atp/"
RAW_WTA = "data/raw/tennis_wta/"
PROCESSED_DIR = "data/processed/"
MODELS_DIR = "models/"
ODDS_DIR = "data/odds/"

# ELO
ELO_START = 1500
ELO_K_BASE = 32
TOURNAMENT_WEIGHTS = {"G": 1.1, "M": 1.0, "A": 0.9, "C": 0.85, "D": 0.7, "F": 1.0}
SURFACE_LIST = ["Hard", "Clay", "Grass", "Carpet"]

# Model
CATBOOST_WEIGHT = 0.6
XGBOOST_WEIGHT = 0.4
OPTUNA_TRIALS = 50
TRAIN_CUTOFF = "2025-01-01"

# Value Betting
MIN_EDGE_THRESHOLD = 0.05  # 5%
MAX_DAILY_BETS = 3
MAX_DAILY_CAPITAL_PCT = 0.50  # 50%
MIN_BET_AMOUNT = 0.50  # €0.50

# Bankroll
DEFAULT_CAPITAL = 5.00

# Scraping
FLASHSCORE_DELAY = 3  # seconds between requests
FUZZY_MATCH_THRESHOLD = 85
```

---

## CODEX TODO (`codex_todo.md`)

```markdown
# CODEX TODO — TennisBet Build

## Phase 1: Data Foundation
- [ ] Clone/download JeffSackmann/tennis_atp into data/raw/tennis_atp/ (git submodule)
- [ ] Clone/download JeffSackmann/tennis_wta into data/raw/tennis_wta/ (git submodule)
- [ ] Create data/custom/, data/meta/, data/odds/ directories with empty starter files
- [ ] Build config.py with all constants
- [ ] Build data_updater.py — git pull primary, HTTP fallback, staleness check per Data Lifecycle spec
- [ ] Implement custom match entry: Streamlit form + CSV upload, saved to data/custom/
- [ ] Implement Flashscore results scraper (completed match results as tertiary data source)
- [ ] Build data_pipeline.py — full implementation per Module 1 spec, merge custom matches
- [ ] Run pipeline, validate outputs, commit processed CSVs
- [ ] Build elo_engine.py — full implementation per Module 2 spec (incremental-capable)
- [ ] Run ELO computation, validate chronological ordering, commit ELO CSVs
- [ ] Initialize data/meta/last_update.json with first run metadata

## Phase 2: Feature Engineering & Models
- [ ] Build feature_engineering.py — all feature groups per Module 3 spec
- [ ] Build tournament_country.json mapping (tournament name → ISO country code)
- [ ] Validate player IOC codes from Sackmann player.csv for home country features
- [ ] Validate no data leakage (spot-check 10 random matches)
- [ ] Build model_training.py — CatBoost + XGBoost per Module 4 spec
- [ ] Run Optuna hyperparameter tuning (50 trials each model, both tours)
- [ ] Train final models, save to models/ directory
- [ ] Log test set metrics: accuracy, log-loss, calibration

## Phase 3: Prediction & Value Engine
- [ ] Build predictor.py — ensemble logic per Module 5 spec
- [ ] Build odds_scraper.py — Flashscore scraper per Module 6 spec
- [ ] Build value_engine.py — value detection + allocation per Module 7 spec
- [ ] Build backtest.py — 3-month backtest per Module 8 spec
- [ ] Run backtest, save results, validate ROI calculation

## Phase 4: Streamlit App
- [ ] Build app.py — all 3 tabs per Streamlit spec
- [ ] Wire up sidebar controls (capital, tour filter, manual odds upload)
- [ ] Implement "Update Data" button in sidebar (triggers data_updater.py)
- [ ] Implement staleness warning banner (color-coded per Data Lifecycle thresholds)
- [ ] Implement "Add Recent Match" custom entry form in sidebar
- [ ] Implement "Refresh odds" flow
- [ ] Implement bankroll tracking: result input after bet day → updates bankroll_log.json
- [ ] Implement prediction_log.csv: log every prediction + actual outcome for performance tracking
- [ ] Test full end-to-end: update → scrape → predict → recommend → display
- [ ] Build requirements.txt with pinned versions

## Phase 5: Polish
- [ ] Add error handling throughout (network failures, missing data, scraper blocks)
- [ ] Add logging (Python logging module, file + console)
- [ ] Write README.md with setup instructions
- [ ] Final test with real upcoming matches

## Phase 6: GitHub Repo + Daily Email Automation
- [ ] Initialize git repo in bet/ folder, create .gitignore (exclude data/raw/, models/*.cbm, __pycache__, .env)
- [ ] Create private GitHub repo (e.g., roman-tennisbet), push initial codebase
- [ ] Create src/daily_report.py — headless script (no Streamlit) that runs the full cycle:
      1. Pull Sackmann data (git pull + HTTP fallback)
      2. Run incremental pipeline (new matches only)
      3. Scrape today's odds from Flashscore
      4. Run ensemble predictor
      5. Run value engine
      6. Generate HTML email body with recommendations (or "SKIP TODAY")
      7. Send email via SMTP (Gmail App Password or SendGrid)
- [ ] Create .github/workflows/daily_bet.yml:
      ```yaml
      name: Daily Tennis Bet Report
      on:
        schedule:
          - cron: '0 7 * * *'   # 7:00 UTC = 9:00 CET (before most ATP matches)
        workflow_dispatch:        # Manual trigger button
      jobs:
        predict:
          runs-on: ubuntu-latest
          steps:
            - uses: actions/checkout@v4
            - uses: actions/setup-python@v5
              with:
                python-version: '3.11'
            - name: Install Chrome + dependencies
              run: |
                sudo apt-get update && sudo apt-get install -y chromium-browser
                pip install -r requirements.txt
            - name: Run daily report
              env:
                EMAIL_TO: ${{ secrets.EMAIL_TO }}
                EMAIL_FROM: ${{ secrets.EMAIL_FROM }}
                EMAIL_PASSWORD: ${{ secrets.EMAIL_PASSWORD }}
                SMTP_HOST: ${{ secrets.SMTP_HOST }}
              run: python src/daily_report.py
            - name: Upload prediction log
              uses: actions/upload-artifact@v4
              with:
                name: prediction-log
                path: data/meta/prediction_log.csv
      ```
- [ ] Create .env.example documenting required secrets
- [ ] Test workflow with workflow_dispatch (manual trigger)
- [ ] Verify email arrives with correct formatting and recommendations
```

---

## CLAUDE TODO (`claude_todo.md`)

```markdown
# CLAUDE TODO — TennisBet QA & Strategic Review

## Phase 1 Review: Data Foundation
- [ ] Review data_updater.py: git pull → HTTP fallback logic, timeout handling, error messages
- [ ] Verify staleness thresholds: are 3/7/14 day cutoffs reasonable for tennis calendar?
- [ ] Review custom match entry: does it match Sackmann schema exactly? Is source="custom" flag preserved through pipeline?
- [ ] Test HTTP fallback: manually break git path, confirm CSV download works
- [ ] Review data_pipeline.py: schema correctness, missing value handling, retirement flagging
- [ ] Verify custom matches merge correctly into master (no duplicates, proper ordering)
- [ ] Review elo_engine.py: verify chronological computation, no future data leakage
- [ ] Verify incremental ELO: does resuming from last state produce same result as full recompute?
- [ ] Spot-check ELO outputs: do top-ranked players have highest ELOs? Surface ELOs diverge logically?
- [ ] Validate processed CSVs: row counts, date ranges, no duplicates

## Phase 2 Review: Features & Models
- [ ] Review feature_engineering.py: confirm ALL features use only pre-match data
- [ ] Verify streak features reset correctly across tournaments and seasons
- [ ] Verify home country mapping: spot-check 10 tournaments for correct country assignment
- [ ] Check home_win_pct: players with <5 home matches should get neutral value (not extreme pct)
- [ ] Leakage audit: verify rolling windows don't include current match
- [ ] Review model_training.py: temporal split is correct, no shuffle in train/test
- [ ] Validate Optuna results: are hyperparameters reasonable? No degenerate solutions?
- [ ] Review feature importances: do they make intuitive sense? Flag any suspicious top features
- [ ] Check calibration: are predicted 60% outcomes actually winning ~60%?

## Phase 3 Review: Prediction & Value
- [ ] Review predictor.py: ensemble weights applied correctly
- [ ] Review odds_scraper.py: player name matching accuracy, rate limiting present
- [ ] Review value_engine.py: overround removal math, allocation caps enforced
- [ ] Review backtest.py: no look-ahead bias, bankroll simulation math correct
- [ ] Validate backtest results: is ROI realistic (not suspiciously high)?

## Phase 4 Review: App
- [ ] Test Streamlit app end-to-end
- [ ] Test "Update Data" button: does staleness banner appear/clear correctly?
- [ ] Test custom match entry form: submit 3 fake matches, verify they appear in predictions
- [ ] Test with stale data (set last_update.json to 15 days ago): does 🔴 critical warning show?
- [ ] Verify bankroll_log.json persists across app restarts
- [ ] Verify prediction_log.csv records every prediction with timestamps
- [ ] Verify "SKIP TODAY" logic triggers correctly when no value
- [ ] Verify allocation percentages sum correctly and respect caps
- [ ] Check edge cases: what if only WTA matches available? Only 1 match today? No matches at all?
- [ ] UX review: are recommendations clear enough for quick decision-making?

## Strategic Checks
- [ ] Compare CatBoost vs XGBoost disagreement rate — if >30% of bets, ensemble weighting needs revision
- [ ] If backtest ROI < 0%: flag for model revision before going live
- [ ] If backtest ROI > 20%: flag as potentially overfit, investigate
- [ ] Verify WTA model isn't significantly weaker than ATP (less data) — may need adjusted thresholds

## Phase 6 Review: GitHub Actions + Email
- [ ] Review daily_report.py: does it run fully headless without Streamlit dependency?
- [ ] Review email HTML template: is it readable on mobile? Does it include all key info (match, edge, stake)?
- [ ] Verify .gitignore: no secrets, no raw data, no model binaries in repo (or use Git LFS for models)
- [ ] Review GitHub Action workflow: does cron timing match CET morning? Is Chrome installed correctly?
- [ ] Security audit: are all secrets in GitHub Secrets, not hardcoded?
- [ ] Test failure mode: what happens if Flashscore scraping fails in CI? Does email still send with "scraper failed" notice?
- [ ] Verify prediction_log.csv artifact upload works for auditability
```

---

## CRITICAL IMPLEMENTATION NOTES

1. **Data Freshness**: Fully handled by `data_updater.py` and the Data Lifecycle spec. Git pull primary, HTTP fallback, staleness banners at 3/7/14 day thresholds, custom match entry as self-sufficiency mechanism. The app shows "data last updated: [date]" in the sidebar at all times.

2. **WTA Data Quality**: WTA has less historical data and more player turnover. The WTA model may need a higher MIN_EDGE_THRESHOLD (7-8%) to compensate for lower confidence. Claude should evaluate this during review.

3. **Player Name Resolution**: This is the #1 breakage point. Sackmann uses full names + IDs, Flashscore uses abbreviated names. The fuzzy matcher MUST have a manual override dictionary for known mismatches. Include a `name_overrides.json` file.

4. **Odds Timing**: Odds move. The scraper should capture odds timestamp and the app should warn if odds were captured >4 hours before match start.

5. **No Live Betting**: This system is for pre-match betting only. Do not attempt in-play predictions.

6. **Bankroll Tracking**: The app should persist capital balance across sessions using a simple JSON file (`bankroll_log.json`). After each betting day, user inputs results, capital updates.

7. **Legal Note**: Display a disclaimer in the app: "This is a statistical model for educational purposes. Past performance does not guarantee future results. Gamble responsibly."

---

## TECH STACK & REQUIREMENTS

```
python>=3.10
streamlit>=1.30
catboost>=1.2
xgboost>=2.0
optuna>=3.5
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
selenium>=4.15  # or playwright
fuzzywuzzy>=0.18
python-Levenshtein>=0.23
plotly>=5.18
requests>=2.31
```

---

## EXECUTION ORDER

1. Codex: Phase 1 (data + ELO) → Claude: Phase 1 review → Sign-off
2. Codex: Phase 2 (features + models) → Claude: Phase 2 review → Sign-off
3. Codex: Phase 3 (prediction + value + backtest) → Claude: Phase 3 review → Sign-off
4. Codex: Phase 4 (Streamlit app) → Claude: Phase 4 review + strategic checks → Sign-off
5. Codex: Phase 5 (polish) → Claude: Final approval → **SHIP IT (local)**
6. Codex: Phase 6 (GitHub repo + Actions + daily email) → Claude: Phase 6 review → **FULLY AUTOMATED**
