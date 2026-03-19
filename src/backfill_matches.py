"""
Backfill missing 2025 + early 2026 ATP/WTA match data from Flashscore.

Outputs Sackmann-format CSVs to data/raw/tennis_atp/ and data/raw/tennis_wta/.

Usage:
    python -m src.backfill_matches --tour atp --year 2025
    python -m src.backfill_matches --tour wta --year 2025
    python -m src.backfill_matches --tour atp --year 2026
    python -m src.backfill_matches --tour wta --year 2026
    python -m src.backfill_matches --all
    python -m src.backfill_matches --tournament australian-open --tour atp --year 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd

# Project imports
from config import RAW_ATP, RAW_WTA

# Reuse player resolution from data_updater
from src.data_updater import (
    _build_player_lookup,
    _load_name_overrides,
    _normalize_name,
    _resolve_player,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RATE_LIMIT_MIN = 2.0
RATE_LIMIT_MAX = 3.5

SACKMANN_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num", "winner_id", "winner_seed", "winner_entry",
    "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand",
    "loser_ht", "loser_ioc", "loser_age", "score", "best_of", "round",
    "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt",
    "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
]

ROUND_ORDER = {
    "F": 300, "SF": 200, "QF": 150, "R16": 100,
    "R32": 50, "R64": 25, "R128": 1, "RR": 175,
    "BR": 175,  # bronze match
}

RAW_DIRS = {"atp": RAW_ATP, "wta": RAW_WTA}

# ---------------------------------------------------------------------------
# Tournament Registry
# ---------------------------------------------------------------------------
# Each entry: {
#   "slug": Flashscore slug (used in URL),
#   "name": Sackmann-style tourney name,
#   "surface": Hard/Clay/Grass,
#   "draw_size": int,
#   "level": G/M/A/F,
#   "best_of": 3 or 5 (for ATP slams),
#   "tourney_date": YYYYMMDD (approximate start date),
#   "tourney_id": Sackmann-style ID,
# }

ATP_2025_TOURNAMENTS = [
    # Grand Slams (best_of=5, level=G, draw_size=128)
    {"slug": "australian-open", "name": "Australian Open", "surface": "Hard", "draw_size": 128, "level": "G", "best_of": 5, "tourney_date": "20250112", "tourney_id": "2025-580"},
    {"slug": "roland-garros", "name": "Roland Garros", "surface": "Clay", "draw_size": 128, "level": "G", "best_of": 5, "tourney_date": "20250525", "tourney_id": "2025-520"},
    {"slug": "wimbledon", "name": "Wimbledon", "surface": "Grass", "draw_size": 128, "level": "G", "best_of": 5, "tourney_date": "20250630", "tourney_id": "2025-540"},
    {"slug": "us-open", "name": "US Open", "surface": "Hard", "draw_size": 128, "level": "G", "best_of": 5, "tourney_date": "20250825", "tourney_id": "2025-560"},
    # Masters 1000 (best_of=3, level=M)
    {"slug": "indian-wells", "name": "Indian Wells Masters", "surface": "Hard", "draw_size": 96, "level": "M", "best_of": 3, "tourney_date": "20250305", "tourney_id": "2025-404"},
    {"slug": "miami", "name": "Miami Masters", "surface": "Hard", "draw_size": 96, "level": "M", "best_of": 3, "tourney_date": "20250319", "tourney_id": "2025-403"},
    {"slug": "monte-carlo", "name": "Monte Carlo Masters", "surface": "Clay", "draw_size": 56, "level": "M", "best_of": 3, "tourney_date": "20250406", "tourney_id": "2025-410"},
    {"slug": "madrid", "name": "Madrid Masters", "surface": "Clay", "draw_size": 56, "level": "M", "best_of": 3, "tourney_date": "20250425", "tourney_id": "2025-1536"},
    {"slug": "rome", "name": "Rome Masters", "surface": "Clay", "draw_size": 56, "level": "M", "best_of": 3, "tourney_date": "20250511", "tourney_id": "2025-416"},
    {"slug": "canada", "name": "Canada Masters", "surface": "Hard", "draw_size": 56, "level": "M", "best_of": 3, "tourney_date": "20250809", "tourney_id": "2025-421"},
    {"slug": "cincinnati", "name": "Cincinnati Masters", "surface": "Hard", "draw_size": 56, "level": "M", "best_of": 3, "tourney_date": "20250816", "tourney_id": "2025-422"},
    {"slug": "shanghai", "name": "Shanghai Masters", "surface": "Hard", "draw_size": 96, "level": "M", "best_of": 3, "tourney_date": "20251004", "tourney_id": "2025-5014"},
    {"slug": "paris", "name": "Paris Masters", "surface": "Hard", "draw_size": 56, "level": "M", "best_of": 3, "tourney_date": "20251027", "tourney_id": "2025-352"},
    # ATP 500 (best_of=3, level=A)
    {"slug": "brisbane", "name": "Brisbane", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250101", "tourney_id": "2025-0339"},
    {"slug": "rotterdam", "name": "Rotterdam", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250203", "tourney_id": "2025-407"},
    {"slug": "rio-de-janeiro", "name": "Rio de Janeiro", "surface": "Clay", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250215", "tourney_id": "2025-6932"},
    {"slug": "acapulco", "name": "Acapulco", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250224", "tourney_id": "2025-7161"},
    {"slug": "dubai", "name": "Dubai", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250224", "tourney_id": "2025-495"},
    {"slug": "barcelona", "name": "Barcelona", "surface": "Clay", "draw_size": 48, "level": "A", "best_of": 3, "tourney_date": "20250414", "tourney_id": "2025-425"},
    {"slug": "hamburg", "name": "Hamburg", "surface": "Clay", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250714", "tourney_id": "2025-414"},
    {"slug": "halle", "name": "Halle", "surface": "Grass", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250616", "tourney_id": "2025-500"},
    {"slug": "queens-club", "name": "Queens Club", "surface": "Grass", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250616", "tourney_id": "2025-311"},
    {"slug": "washington", "name": "Washington", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250728", "tourney_id": "2025-418"},
    {"slug": "tokyo", "name": "Tokyo", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250920", "tourney_id": "2025-329"},
    {"slug": "basel", "name": "Basel", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20251020", "tourney_id": "2025-328"},
    {"slug": "vienna", "name": "Vienna", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20251020", "tourney_id": "2025-337"},
    # ATP 250s
    {"slug": "adelaide", "name": "Adelaide", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250106", "tourney_id": "2025-8998"},
    {"slug": "auckland", "name": "Auckland", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250106", "tourney_id": "2025-301"},
    {"slug": "hong-kong", "name": "Hong Kong", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250101", "tourney_id": "2025-9426"},
    {"slug": "montpellier", "name": "Montpellier", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250203", "tourney_id": "2025-375"},
    {"slug": "dallas", "name": "Dallas", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250203", "tourney_id": "2025-9158"},
    {"slug": "marseille", "name": "Marseille", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250217", "tourney_id": "2025-496"},
    {"slug": "delray-beach", "name": "Delray Beach", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250217", "tourney_id": "2025-499"},
    {"slug": "buenos-aires", "name": "Buenos Aires", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250210", "tourney_id": "2025-506"},
    {"slug": "santiago", "name": "Santiago", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250224", "tourney_id": "2025-7485"},
    {"slug": "doha", "name": "Doha", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250203", "tourney_id": "2025-451"},
    {"slug": "marrakech", "name": "Marrakech", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250407", "tourney_id": "2025-308"},
    {"slug": "bucharest", "name": "Bucharest", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250414", "tourney_id": "2025-4120"},
    {"slug": "munich", "name": "Munich", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250421", "tourney_id": "2025-308"},
    {"slug": "lyon", "name": "Lyon", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250519", "tourney_id": "2025-7694"},
    {"slug": "geneva", "name": "Geneva", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250519", "tourney_id": "2025-322"},
    {"slug": "s-hertogenbosch", "name": "s-Hertogenbosch", "surface": "Grass", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250609", "tourney_id": "2025-440"},
    {"slug": "stuttgart", "name": "Stuttgart", "surface": "Grass", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250609", "tourney_id": "2025-321"},
    {"slug": "eastbourne", "name": "Eastbourne", "surface": "Grass", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250623", "tourney_id": "2025-741"},
    {"slug": "mallorca", "name": "Mallorca", "surface": "Grass", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250623", "tourney_id": "2025-7434"},
    {"slug": "bastad", "name": "Bastad", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250714", "tourney_id": "2025-316"},
    {"slug": "umag", "name": "Umag", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250721", "tourney_id": "2025-7290"},
    {"slug": "atlanta", "name": "Atlanta", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250721", "tourney_id": "2025-319"},
    {"slug": "gstaad", "name": "Gstaad", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250721", "tourney_id": "2025-314"},
    {"slug": "kitzbuhel", "name": "Kitzbuhel", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250728", "tourney_id": "2025-7480"},
    {"slug": "los-cabos", "name": "Los Cabos", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250728", "tourney_id": "2025-7696"},
    {"slug": "winston-salem", "name": "Winston-Salem", "surface": "Hard", "draw_size": 48, "level": "A", "best_of": 3, "tourney_date": "20250818", "tourney_id": "2025-7481"},
    {"slug": "chengdu", "name": "Chengdu", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250922", "tourney_id": "2025-7581"},
    {"slug": "beijing", "name": "Beijing", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250927", "tourney_id": "2025-747"},
    {"slug": "almaty", "name": "Almaty", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250922", "tourney_id": "2025-9410"},
    {"slug": "antwerp", "name": "Antwerp", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20251013", "tourney_id": "2025-7485"},
    {"slug": "stockholm", "name": "Stockholm", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20251013", "tourney_id": "2025-429"},
    {"slug": "metz", "name": "Metz", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20251027", "tourney_id": "2025-341"},
    {"slug": "belgrade", "name": "Belgrade", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20251103", "tourney_id": "2025-9158"},
    # Tour Finals
    {"slug": "atp-finals", "name": "Tour Finals", "surface": "Hard", "draw_size": 8, "level": "F", "best_of": 3, "tourney_date": "20251109", "tourney_id": "2025-605"},
]

ATP_2026_TOURNAMENTS = [
    # Grand Slams
    {"slug": "australian-open", "name": "Australian Open", "surface": "Hard", "draw_size": 128, "level": "G", "best_of": 5, "tourney_date": "20260119", "tourney_id": "2026-580"},
    # Early season
    {"slug": "brisbane", "name": "Brisbane", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20260101", "tourney_id": "2026-0339"},
    {"slug": "hong-kong", "name": "Hong Kong", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260101", "tourney_id": "2026-9426"},
    {"slug": "adelaide", "name": "Adelaide", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260106", "tourney_id": "2026-8998"},
    {"slug": "auckland", "name": "Auckland", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260106", "tourney_id": "2026-301"},
    {"slug": "montpellier", "name": "Montpellier", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260202", "tourney_id": "2026-375"},
    {"slug": "dallas", "name": "Dallas", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260202", "tourney_id": "2026-9158"},
    {"slug": "rotterdam", "name": "Rotterdam", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20260209", "tourney_id": "2026-407"},
    {"slug": "doha", "name": "Doha", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260209", "tourney_id": "2026-451"},
    {"slug": "buenos-aires", "name": "Buenos Aires", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260209", "tourney_id": "2026-506"},
    {"slug": "marseille", "name": "Marseille", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260216", "tourney_id": "2026-496"},
    {"slug": "rio-de-janeiro", "name": "Rio de Janeiro", "surface": "Clay", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20260216", "tourney_id": "2026-6932"},
    {"slug": "delray-beach", "name": "Delray Beach", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260216", "tourney_id": "2026-499"},
    {"slug": "acapulco", "name": "Acapulco", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20260223", "tourney_id": "2026-7161"},
    {"slug": "dubai", "name": "Dubai", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20260223", "tourney_id": "2026-495"},
    {"slug": "santiago", "name": "Santiago", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260223", "tourney_id": "2026-7485"},
    # Masters
    {"slug": "indian-wells", "name": "Indian Wells Masters", "surface": "Hard", "draw_size": 96, "level": "M", "best_of": 3, "tourney_date": "20260305", "tourney_id": "2026-404"},
]

WTA_2025_TOURNAMENTS = [
    # Grand Slams (best_of=3, level=G, draw_size=128)
    {"slug": "australian-open", "name": "Australian Open", "surface": "Hard", "draw_size": 128, "level": "G", "best_of": 3, "tourney_date": "20250112", "tourney_id": "2025-580"},
    {"slug": "roland-garros", "name": "Roland Garros", "surface": "Clay", "draw_size": 128, "level": "G", "best_of": 3, "tourney_date": "20250525", "tourney_id": "2025-520"},
    {"slug": "wimbledon", "name": "Wimbledon", "surface": "Grass", "draw_size": 128, "level": "G", "best_of": 3, "tourney_date": "20250630", "tourney_id": "2025-540"},
    {"slug": "us-open", "name": "US Open", "surface": "Hard", "draw_size": 128, "level": "G", "best_of": 3, "tourney_date": "20250825", "tourney_id": "2025-560"},
    # WTA 1000
    {"slug": "indian-wells", "name": "Indian Wells", "surface": "Hard", "draw_size": 96, "level": "M", "best_of": 3, "tourney_date": "20250305", "tourney_id": "2025-404"},
    {"slug": "miami", "name": "Miami", "surface": "Hard", "draw_size": 96, "level": "M", "best_of": 3, "tourney_date": "20250319", "tourney_id": "2025-403"},
    {"slug": "madrid", "name": "Madrid", "surface": "Clay", "draw_size": 64, "level": "M", "best_of": 3, "tourney_date": "20250425", "tourney_id": "2025-1536"},
    {"slug": "rome", "name": "Rome", "surface": "Clay", "draw_size": 64, "level": "M", "best_of": 3, "tourney_date": "20250511", "tourney_id": "2025-416"},
    {"slug": "canada", "name": "Canada", "surface": "Hard", "draw_size": 64, "level": "M", "best_of": 3, "tourney_date": "20250809", "tourney_id": "2025-421"},
    {"slug": "cincinnati", "name": "Cincinnati", "surface": "Hard", "draw_size": 64, "level": "M", "best_of": 3, "tourney_date": "20250816", "tourney_id": "2025-422"},
    {"slug": "beijing", "name": "Beijing", "surface": "Hard", "draw_size": 64, "level": "M", "best_of": 3, "tourney_date": "20250927", "tourney_id": "2025-747"},
    {"slug": "wuhan", "name": "Wuhan", "surface": "Hard", "draw_size": 64, "level": "M", "best_of": 3, "tourney_date": "20251019", "tourney_id": "2025-1054"},
    # WTA 500
    {"slug": "brisbane", "name": "Brisbane", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250101", "tourney_id": "2025-0339"},
    {"slug": "abu-dhabi", "name": "Abu Dhabi", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250203", "tourney_id": "2025-9409"},
    {"slug": "doha", "name": "Doha", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250210", "tourney_id": "2025-451"},
    {"slug": "dubai", "name": "Dubai", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20250217", "tourney_id": "2025-495"},
    {"slug": "san-diego", "name": "San Diego", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250302", "tourney_id": "2025-9158"},
    {"slug": "berlin", "name": "Berlin", "surface": "Grass", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250616", "tourney_id": "2025-2014"},
    {"slug": "eastbourne", "name": "Eastbourne", "surface": "Grass", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250623", "tourney_id": "2025-741"},
    {"slug": "tokyo", "name": "Tokyo", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250920", "tourney_id": "2025-329"},
    {"slug": "seoul", "name": "Seoul", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250922", "tourney_id": "2025-9326"},
    # WTA 250
    {"slug": "adelaide", "name": "Adelaide", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250106", "tourney_id": "2025-8998"},
    {"slug": "auckland", "name": "Auckland", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250106", "tourney_id": "2025-301"},
    {"slug": "hobart", "name": "Hobart", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250106", "tourney_id": "2025-352"},
    {"slug": "linz", "name": "Linz", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250127", "tourney_id": "2025-7694"},
    {"slug": "singapore", "name": "Singapore", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250127", "tourney_id": "2025-7695"},
    {"slug": "austin", "name": "Austin", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250224", "tourney_id": "2025-7696"},
    {"slug": "monterrey", "name": "Monterrey", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250310", "tourney_id": "2025-7697"},
    {"slug": "lyon", "name": "Lyon", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250302", "tourney_id": "2025-7694"},
    {"slug": "charleston", "name": "Charleston", "surface": "Clay", "draw_size": 56, "level": "A", "best_of": 3, "tourney_date": "20250330", "tourney_id": "2025-421"},
    {"slug": "bogota", "name": "Bogota", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250407", "tourney_id": "2025-7698"},
    {"slug": "rouen", "name": "Rouen", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250414", "tourney_id": "2025-7699"},
    {"slug": "rabat", "name": "Rabat", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250421", "tourney_id": "2025-7700"},
    {"slug": "strasbourg", "name": "Strasbourg", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250519", "tourney_id": "2025-337"},
    {"slug": "nottingham", "name": "Nottingham", "surface": "Grass", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250609", "tourney_id": "2025-741"},
    {"slug": "birmingham", "name": "Birmingham", "surface": "Grass", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250616", "tourney_id": "2025-742"},
    {"slug": "bad-homburg", "name": "Bad Homburg", "surface": "Grass", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250623", "tourney_id": "2025-743"},
    {"slug": "budapest", "name": "Budapest", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250714", "tourney_id": "2025-744"},
    {"slug": "iasi", "name": "Iasi", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250714", "tourney_id": "2025-745"},
    {"slug": "prague", "name": "Prague", "surface": "Clay", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250721", "tourney_id": "2025-746"},
    {"slug": "washington", "name": "Washington", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250728", "tourney_id": "2025-418"},
    {"slug": "monterrey-ii", "name": "Monterrey II", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250901", "tourney_id": "2025-7701"},
    {"slug": "guadalajara", "name": "Guadalajara", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20250908", "tourney_id": "2025-7702"},
    {"slug": "ningbo", "name": "Ningbo", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20251004", "tourney_id": "2025-7703"},
    {"slug": "osaka", "name": "Osaka", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20251011", "tourney_id": "2025-7704"},
    {"slug": "hong-kong", "name": "Hong Kong", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20251027", "tourney_id": "2025-7705"},
    # Tour Finals
    {"slug": "wta-finals", "name": "WTA Finals", "surface": "Hard", "draw_size": 8, "level": "F", "best_of": 3, "tourney_date": "20251103", "tourney_id": "2025-605"},
]

WTA_2026_TOURNAMENTS = [
    # Grand Slam
    {"slug": "australian-open", "name": "Australian Open", "surface": "Hard", "draw_size": 128, "level": "G", "best_of": 3, "tourney_date": "20260119", "tourney_id": "2026-580"},
    # Early season
    {"slug": "brisbane", "name": "Brisbane", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20260101", "tourney_id": "2026-0339"},
    {"slug": "adelaide", "name": "Adelaide", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260106", "tourney_id": "2026-8998"},
    {"slug": "auckland", "name": "Auckland", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260106", "tourney_id": "2026-301"},
    {"slug": "hobart", "name": "Hobart", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260112", "tourney_id": "2026-352"},
    {"slug": "abu-dhabi", "name": "Abu Dhabi", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260202", "tourney_id": "2026-9409"},
    {"slug": "linz", "name": "Linz", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260202", "tourney_id": "2026-7694"},
    {"slug": "doha", "name": "Doha", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260209", "tourney_id": "2026-451"},
    {"slug": "dubai", "name": "Dubai", "surface": "Hard", "draw_size": 32, "level": "A", "best_of": 3, "tourney_date": "20260216", "tourney_id": "2026-495"},
    {"slug": "san-diego", "name": "San Diego", "surface": "Hard", "draw_size": 28, "level": "A", "best_of": 3, "tourney_date": "20260302", "tourney_id": "2026-9158"},
    # WTA 1000
    {"slug": "indian-wells", "name": "Indian Wells", "surface": "Hard", "draw_size": 96, "level": "M", "best_of": 3, "tourney_date": "20260305", "tourney_id": "2026-404"},
]

ALL_REGISTRIES = {
    ("atp", 2025): ATP_2025_TOURNAMENTS,
    ("atp", 2026): ATP_2026_TOURNAMENTS,
    ("wta", 2025): WTA_2025_TOURNAMENTS,
    ("wta", 2026): WTA_2026_TOURNAMENTS,
}


# ---------------------------------------------------------------------------
# Selenium Driver
# ---------------------------------------------------------------------------
def _init_driver():
    """Initialize headless Chrome with anti-detection flags."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=en-US")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    chrome_bin = os.environ.get("CHROME_BIN") or os.environ.get("GOOGLE_CHROME_BIN")
    chromedriver_path = os.environ.get("CHROMEDRIVER_PATH")
    if chrome_bin:
        options.binary_location = chrome_bin

    service = Service(executable_path=chromedriver_path) if chromedriver_path else None
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(5)
    return driver


def _sleep():
    """Rate-limiting sleep between page loads."""
    time.sleep(random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX))


# ---------------------------------------------------------------------------
# Player Metadata
# ---------------------------------------------------------------------------
def _load_player_metadata(tour: str) -> dict[str, dict[str, Any]]:
    """Load player CSV into {player_id_str: {hand, ht, ioc, dob}} dict."""
    path = RAW_DIRS[tour] / f"{tour}_players.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    meta: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        try:
            pid = str(int(float(row["player_id"])))
        except (ValueError, TypeError):
            continue
        meta[pid] = {
            "hand": str(row.get("hand", "")) if pd.notna(row.get("hand")) else "",
            "ht": int(float(row["height"])) if pd.notna(row.get("height")) else "",
            "ioc": str(row.get("ioc", "")) if pd.notna(row.get("ioc")) else "",
            "dob": int(float(row["dob"])) if pd.notna(row.get("dob")) else None,
        }
    return meta


def _compute_age(dob_int: int | None, tourney_date_str: str) -> str:
    """Compute player age from DOB (YYYYMMDD int) and tourney_date (YYYYMMDD str)."""
    if not dob_int:
        return ""
    try:
        dob = datetime.strptime(str(int(dob_int)), "%Y%m%d")
        tdate = datetime.strptime(tourney_date_str, "%Y%m%d")
        age = round((tdate - dob).days / 365.25, 1)
        return str(age) if age > 10 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Flashscore Round Detection
# ---------------------------------------------------------------------------
ROUND_PATTERNS = {
    "Final": "F",
    "Semi-final": "SF",
    "Semi-finals": "SF",
    "Quarter-final": "QF",
    "Quarter-finals": "QF",
    "Round of 16": "R16",
    "8th-finals": "R16",
    "Round of 32": "R32",
    "Round of 64": "R64",
    "Round of 128": "R128",
    "1st Round": "R128",  # context-dependent, will be adjusted by draw size
    "2nd Round": "R64",
    "3rd Round": "R32",
    "4th Round": "R16",
    "1st round": "R128",
    "2nd round": "R64",
    "3rd round": "R32",
    "4th round": "R16",
    "Round Robin": "RR",
    "Group": "RR",
    "3rd place": "BR",
}


def _detect_round(text: str, draw_size: int = 128) -> str:
    """Detect round code from Flashscore round header text."""
    text_clean = text.strip()
    for pattern, code in ROUND_PATTERNS.items():
        if pattern.lower() in text_clean.lower():
            # Adjust "1st Round" etc. based on draw size
            if code == "R128" and draw_size <= 32:
                return "R32"
            elif code == "R128" and draw_size <= 64:
                return "R64"
            if code == "R64" and draw_size <= 32:
                return "R16"
            if code == "R32" and draw_size <= 32:
                return "QF"
            return code
    return "R32"  # fallback


def _adjust_rounds_for_draw_size(round_code: str, draw_size: int) -> str:
    """Map generic numbered rounds to Sackmann round codes based on draw size."""
    return round_code  # Already handled in _detect_round


# ---------------------------------------------------------------------------
# Score Parsing from Flashscore
# ---------------------------------------------------------------------------
def _parse_score_from_row(match_el) -> tuple[str | None, int]:
    """Extract score string and set count from a Flashscore match element."""
    home_parts = match_el.find_elements("css selector", "div.event__part--home")
    away_parts = match_el.find_elements("css selector", "div.event__part--away")
    n = min(len(home_parts), len(away_parts))
    if n == 0:
        return None, 0
    sets = []
    for i in range(n):
        h = (home_parts[i].text or "").strip()
        a = (away_parts[i].text or "").strip()
        if h.isdigit() and a.isdigit():
            sets.append(f"{h}-{a}")
    if not sets:
        return None, 0
    return " ".join(sets), len(sets)


def _detect_tiebreak_scores(match_el, score_str: str) -> str:
    """Try to detect tiebreak scores from superscript elements. Returns enriched score."""
    # Flashscore sometimes shows tiebreak scores in separate elements
    # For now, return the basic score - tiebreak detection can be enhanced later
    return score_str


# ---------------------------------------------------------------------------
# Main Scraping Logic
# ---------------------------------------------------------------------------
def _click_show_more(driver, max_clicks: int = 20) -> int:
    """Click 'Show more matches' button until no more results. Returns click count."""
    clicks = 0
    for _ in range(max_clicks):
        try:
            show_more = driver.find_elements("css selector", "a.event__more")
            if not show_more:
                show_more = driver.find_elements("xpath", "//a[contains(@class, 'event__more')]")
            if not show_more:
                break
            btn = show_more[0]
            if not btn.is_displayed():
                break
            btn.click()
            clicks += 1
            time.sleep(random.uniform(1.0, 2.0))
        except Exception:
            break
    return clicks


def scrape_tournament(
    driver,
    tournament: dict[str, Any],
    tour: str,
    year: int,
    player_pool: list[str],
    name_to_id: dict[str, str],
    last_initial_idx: dict,
    overrides: dict[str, str],
    player_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scrape all matches for a single tournament from Flashscore."""
    slug = tournament["slug"]
    tour_prefix = "atp-singles" if tour == "atp" else "wta-singles"

    # Try multiple URL patterns
    urls_to_try = [
        f"https://www.flashscore.com/tennis/{tour_prefix}/{slug}/results/?seasonId={year}",
        f"https://www.flashscore.com/tennis/{tour_prefix}/{slug}-{year}/results/",
        f"https://www.flashscore.com/tennis/{tour_prefix}/{slug}/results/",
    ]

    matches: list[dict[str, Any]] = []
    page_loaded = False

    for url in urls_to_try:
        try:
            log.info(f"  Trying: {url}")
            driver.get(url)
            _sleep()

            # Check if page loaded successfully (has match elements or tournament header)
            elements = driver.find_elements("css selector", "div.event__match")
            if elements:
                page_loaded = True
                log.info(f"  Found {len(elements)} match elements on initial load")
                break

            # Try clicking Results tab if needed
            tabs = driver.find_elements("css selector", "a.tabs__tab")
            for tab in tabs:
                if (tab.text or "").strip().upper() in ("RESULTS", "VÝSLEDKY"):
                    tab.click()
                    _sleep()
                    elements = driver.find_elements("css selector", "div.event__match")
                    if elements:
                        page_loaded = True
                        log.info(f"  Found {len(elements)} after clicking Results tab")
                        break
            if page_loaded:
                break
        except Exception as e:
            log.warning(f"  URL failed: {url} - {e}")
            continue

    if not page_loaded:
        log.warning(f"  Could not load results for {tournament['name']} {year}")
        return []

    # Click "Show more" to load all matches
    more_clicks = _click_show_more(driver)
    if more_clicks > 0:
        log.info(f"  Clicked 'Show more' {more_clicks} times")

    # Parse all match rows
    all_elements = driver.find_elements("css selector", "div.event__match, div.event__round")

    current_round = "R32"
    match_counter = 0
    tourney_date = tournament["tourney_date"]
    draw_size = tournament["draw_size"]

    for el in all_elements:
        classes = el.get_attribute("class") or ""

        # Round header
        if "event__round" in classes:
            round_text = (el.text or "").strip()
            if round_text:
                current_round = _detect_round(round_text, draw_size)
            continue

        # Match row
        if "event__match" not in classes:
            continue

        try:
            home_el = el.find_element("css selector", "div.event__participant--home")
            away_el = el.find_element("css selector", "div.event__participant--away")
            home_raw = (home_el.text or "").strip()
            away_raw = (away_el.text or "").strip()

            if not home_raw or not away_raw:
                continue

            # Skip doubles (contains "/")
            if "/" in home_raw or "/" in away_raw:
                continue

            score_text, set_count = _parse_score_from_row(el)
            if not score_text:
                continue

            # Enrich score with tiebreak info
            score_text = _detect_tiebreak_scores(el, score_text)

            # Determine winner
            home_is_winner = "fontExtraBold" in (home_el.get_attribute("class") or "")
            away_is_winner = "fontExtraBold" in (away_el.get_attribute("class") or "")

            if not home_is_winner and not away_is_winner:
                # Try parent/container bold detection
                try:
                    home_parent = home_el.find_element("xpath", "..")
                    away_parent = away_el.find_element("xpath", "..")
                    home_is_winner = "fontExtraBold" in (home_parent.get_attribute("class") or "")
                    away_is_winner = "fontExtraBold" in (away_parent.get_attribute("class") or "")
                except Exception:
                    pass

            if not home_is_winner and not away_is_winner:
                # Fallback: first player listed with higher set count wins
                home_sets = sum(1 for s in score_text.split() if "-" in s and int(s.split("-")[0]) > int(s.split("-")[1]))
                away_sets = sum(1 for s in score_text.split() if "-" in s and int(s.split("-")[1]) > int(s.split("-")[0]))
                home_is_winner = home_sets > away_sets

            # Resolve players
            w_raw = home_raw if home_is_winner else away_raw
            l_raw = away_raw if home_is_winner else home_raw

            w_name, w_id, w_score = _resolve_player(
                w_raw, player_pool, name_to_id, last_initial_idx, overrides
            )
            l_name, l_id, l_score = _resolve_player(
                l_raw, player_pool, name_to_id, last_initial_idx, overrides
            )

            if not w_id or not l_id:
                log.debug(f"  Skipped unresolved: {w_raw} vs {l_raw} (w_id={w_id}, l_id={l_id})")
                continue

            # Player metadata
            w_meta = player_meta.get(w_id, {})
            l_meta = player_meta.get(l_id, {})

            match_counter += 1
            match_num = ROUND_ORDER.get(current_round, 50) + match_counter

            row = {
                "tourney_id": tournament["tourney_id"],
                "tourney_name": tournament["name"],
                "surface": tournament["surface"],
                "draw_size": draw_size,
                "tourney_level": tournament["level"],
                "tourney_date": tourney_date,
                "match_num": match_num,
                "winner_id": w_id,
                "winner_seed": "",
                "winner_entry": "",
                "winner_name": w_name,
                "winner_hand": w_meta.get("hand", ""),
                "winner_ht": w_meta.get("ht", ""),
                "winner_ioc": w_meta.get("ioc", ""),
                "winner_age": _compute_age(w_meta.get("dob"), tourney_date),
                "loser_id": l_id,
                "loser_seed": "",
                "loser_entry": "",
                "loser_name": l_name,
                "loser_hand": l_meta.get("hand", ""),
                "loser_ht": l_meta.get("ht", ""),
                "loser_ioc": l_meta.get("ioc", ""),
                "loser_age": _compute_age(l_meta.get("dob"), tourney_date),
                "score": score_text,
                "best_of": tournament["best_of"],
                "round": current_round,
                "minutes": "",
                "w_ace": "", "w_df": "", "w_svpt": "", "w_1stIn": "",
                "w_1stWon": "", "w_2ndWon": "", "w_SvGms": "",
                "w_bpSaved": "", "w_bpFaced": "",
                "l_ace": "", "l_df": "", "l_svpt": "", "l_1stIn": "",
                "l_1stWon": "", "l_2ndWon": "", "l_SvGms": "",
                "l_bpSaved": "", "l_bpFaced": "",
                "winner_rank": "", "winner_rank_points": "",
                "loser_rank": "", "loser_rank_points": "",
            }
            matches.append(row)

        except Exception as e:
            log.debug(f"  Error parsing match: {e}")
            continue

    log.info(f"  Scraped {len(matches)} matches from {tournament['name']} {year}")
    return matches


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _write_output(
    matches: list[dict[str, Any]],
    tour: str,
    year: int,
    incremental: bool = True,
) -> Path:
    """Write matches to Sackmann-format CSV, optionally merging with existing."""
    raw_dir = RAW_DIRS[tour]
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{tour}_matches_{year}.csv"

    new_df = pd.DataFrame(matches, columns=SACKMANN_COLUMNS)

    if incremental and output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path, low_memory=False)
        log.info(f"  Existing file has {len(existing)} rows")
        # Dedup key: tourney_id + winner_id + loser_id + round
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined["_dedup"] = (
            combined["tourney_id"].astype(str) + "|"
            + combined["winner_id"].astype(str) + "|"
            + combined["loser_id"].astype(str) + "|"
            + combined["round"].astype(str)
        )
        combined = combined.drop_duplicates(subset=["_dedup"], keep="last")
        combined = combined.drop(columns=["_dedup"])
        new_df = combined

    # Sort by tourney_date, tourney_id, match_num
    new_df = new_df.sort_values(
        by=["tourney_date", "tourney_id", "match_num"],
        ascending=True,
    ).reset_index(drop=True)

    new_df.to_csv(output_path, index=False)
    log.info(f"  Written {len(new_df)} rows to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_output(path: Path) -> dict[str, Any]:
    """Run basic validation on output CSV."""
    if not path.exists():
        return {"valid": False, "error": "File does not exist"}

    df = pd.read_csv(path, low_memory=False)
    issues: list[str] = []

    # Check header
    expected = set(SACKMANN_COLUMNS)
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Check for duplicates
    df["_key"] = (
        df["tourney_id"].astype(str) + "|"
        + df["winner_id"].astype(str) + "|"
        + df["loser_id"].astype(str) + "|"
        + df["round"].astype(str)
    )
    dups = df.duplicated(subset=["_key"]).sum()
    if dups > 0:
        issues.append(f"{dups} duplicate matches")

    # Check tourney_date format
    invalid_dates = 0
    for val in df["tourney_date"].dropna().unique():
        try:
            datetime.strptime(str(int(float(val))), "%Y%m%d")
        except Exception:
            invalid_dates += 1
    if invalid_dates > 0:
        issues.append(f"{invalid_dates} invalid tourney_date values")

    # Check score format
    invalid_scores = 0
    for score in df["score"].dropna():
        if not re.match(r"^[\d\-\(\) ]+$", str(score)):
            invalid_scores += 1
    if invalid_scores > 0:
        issues.append(f"{invalid_scores} unusual score formats")

    return {
        "valid": len(issues) == 0,
        "rows": len(df),
        "tournaments": df["tourney_id"].nunique(),
        "issues": issues,
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def backfill(
    tour: str,
    year: int,
    tournament_slug: str | None = None,
    incremental: bool = True,
) -> dict[str, Any]:
    """
    Backfill match data for a tour/year from Flashscore.

    Args:
        tour: "atp" or "wta"
        year: 2025 or 2026
        tournament_slug: Optional specific tournament slug to scrape
        incremental: If True, merge with existing CSV

    Returns:
        Report dict with scraping results.
    """
    registry_key = (tour, year)
    if registry_key not in ALL_REGISTRIES:
        return {"success": False, "error": f"No registry for {tour} {year}"}

    tournaments = ALL_REGISTRIES[registry_key]

    if tournament_slug:
        tournaments = [t for t in tournaments if t["slug"] == tournament_slug]
        if not tournaments:
            return {"success": False, "error": f"Tournament '{tournament_slug}' not found in {tour} {year}"}

    log.info(f"=" * 60)
    log.info(f"Backfilling {tour.upper()} {year}: {len(tournaments)} tournaments")
    log.info(f"=" * 60)

    # Load player resolution infrastructure
    log.info("Loading player lookup...")
    player_pool, name_to_id, last_initial_idx = _build_player_lookup()
    overrides = _load_name_overrides()
    log.info(f"  Player pool: {len(player_pool)} names")

    # Load player metadata for enrichment
    log.info("Loading player metadata...")
    # Load both ATP and WTA metadata for cross-tour resolution
    player_meta: dict[str, dict[str, Any]] = {}
    for t in ("atp", "wta"):
        player_meta.update(_load_player_metadata(t))
    log.info(f"  Player metadata: {len(player_meta)} entries")

    # Initialize Selenium
    log.info("Initializing Chrome driver...")
    driver = _init_driver()

    all_matches: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    try:
        # Handle cookie consent popup if present
        try:
            driver.get("https://www.flashscore.com")
            _sleep()
            consent_buttons = driver.find_elements("css selector", "#onetrust-accept-btn-handler")
            if consent_buttons:
                consent_buttons[0].click()
                time.sleep(1)
                log.info("  Accepted cookie consent")
        except Exception:
            pass

        for i, tournament in enumerate(tournaments):
            log.info(f"\n[{i+1}/{len(tournaments)}] {tournament['name']} ({tournament['surface']}, {tournament['level']})")

            try:
                matches = scrape_tournament(
                    driver=driver,
                    tournament=tournament,
                    tour=tour,
                    year=year,
                    player_pool=player_pool,
                    name_to_id=name_to_id,
                    last_initial_idx=last_initial_idx,
                    overrides=overrides,
                    player_meta=player_meta,
                )
                all_matches.extend(matches)
                results.append({
                    "tournament": tournament["name"],
                    "slug": tournament["slug"],
                    "matches": len(matches),
                    "success": True,
                })
            except Exception as e:
                log.error(f"  Failed: {e}")
                results.append({
                    "tournament": tournament["name"],
                    "slug": tournament["slug"],
                    "matches": 0,
                    "success": False,
                    "error": str(e),
                })

            # Extra delay between tournaments
            if i < len(tournaments) - 1:
                time.sleep(random.uniform(1.0, 2.0))

    finally:
        driver.quit()
        log.info("Chrome driver closed")

    # Write output
    if all_matches:
        output_path = _write_output(all_matches, tour, year, incremental=incremental)
        validation = _validate_output(output_path)
    else:
        output_path = None
        validation = {"valid": False, "rows": 0, "issues": ["No matches scraped"]}

    report = {
        "success": len(all_matches) > 0,
        "tour": tour,
        "year": year,
        "total_matches": len(all_matches),
        "tournaments_attempted": len(tournaments),
        "tournaments_with_data": sum(1 for r in results if r["matches"] > 0),
        "output_file": str(output_path) if output_path else None,
        "validation": validation,
        "tournament_results": results,
    }

    log.info(f"\n{'=' * 60}")
    log.info(f"BACKFILL COMPLETE: {tour.upper()} {year}")
    log.info(f"  Total matches: {len(all_matches)}")
    log.info(f"  Tournaments with data: {report['tournaments_with_data']}/{len(tournaments)}")
    if output_path:
        log.info(f"  Output: {output_path}")
    log.info(f"  Validation: {'PASS' if validation.get('valid') else 'ISSUES: ' + str(validation.get('issues', []))}")
    log.info(f"{'=' * 60}")

    return report


def backfill_all() -> dict[str, Any]:
    """Backfill all tours and years."""
    reports = {}
    for (tour, year) in [("atp", 2025), ("wta", 2025), ("atp", 2026), ("wta", 2026)]:
        key = f"{tour}_{year}"
        log.info(f"\n{'#' * 60}")
        log.info(f"# Starting {key}")
        log.info(f"{'#' * 60}")
        reports[key] = backfill(tour, year)
    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Backfill tennis match data from Flashscore")
    parser.add_argument("--tour", choices=["atp", "wta"], help="Tour to backfill")
    parser.add_argument("--year", type=int, help="Year to backfill")
    parser.add_argument("--tournament", type=str, help="Specific tournament slug")
    parser.add_argument("--all", action="store_true", help="Backfill all tours and years")
    parser.add_argument("--no-incremental", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.all:
        reports = backfill_all()
        print(json.dumps(reports, indent=2, default=str))
        return

    if not args.tour or not args.year:
        parser.error("--tour and --year are required (or use --all)")

    report = backfill(
        tour=args.tour,
        year=args.year,
        tournament_slug=args.tournament,
        incremental=not args.no_incremental,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
