from __future__ import annotations

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_ATP = RAW_DIR / "tennis_atp"
RAW_WTA = RAW_DIR / "tennis_wta"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ODDS_DIR = DATA_DIR / "odds"
CUSTOM_DIR = DATA_DIR / "custom"
META_DIR = DATA_DIR / "meta"

# Core files
LAST_UPDATE_FILE = META_DIR / "last_update.json"
BANKROLL_LOG_FILE = META_DIR / "bankroll_log.json"
PREDICTION_LOG_FILE = META_DIR / "prediction_log.csv"
ODDS_UPCOMING_FILE = ODDS_DIR / "upcoming_odds.csv"
ODDS_HISTORY_FILE = ODDS_DIR / "odds_history.csv"
CUSTOM_ATP_FILE = CUSTOM_DIR / "custom_matches_atp.csv"
CUSTOM_WTA_FILE = CUSTOM_DIR / "custom_matches_wta.csv"

# ELO
ELO_START = 1500.0
ELO_K_BASE = 32.0
TOURNAMENT_WEIGHTS = {
    "G": 1.10,  # Grand Slam
    "M": 1.00,  # Masters
    "A": 0.90,  # ATP/WTA 250/500
    "C": 0.85,  # Challenger/ITF-like in source schema
    "D": 0.70,
    "F": 1.00,
    "I": 1.00,
}
SURFACE_LIST = ["Hard", "Clay", "Grass", "Carpet"]

# Model
CATBOOST_WEIGHT = 0.6
XGBOOST_WEIGHT = 0.4
OPTUNA_TRIALS = 50
TRAIN_CUTOFF = "2025-01-01"

# Value betting
MIN_EDGE_THRESHOLD = 0.05
MAX_DAILY_BETS = 3
MAX_DAILY_CAPITAL_PCT = 0.50
MIN_BET_AMOUNT = 0.50

# Bankroll
DEFAULT_CAPITAL = 5.00

# Scraping/matching
FLASHSCORE_DELAY = 3
FUZZY_MATCH_THRESHOLD = 85

# Data freshness
FRESH_DAYS = 3
WARNING_DAYS = 7
STALE_DAYS = 14

# Misc
DATE_FMT = "%Y-%m-%d"
