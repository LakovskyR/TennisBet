"""
WTA match data backfill using a waterfall of sources:

    1. Jeff Sackmann's tennis_wta GitHub (check first — may be updated)
    2. Tennis Explorer scraping via Firecrawl API
    3. Perplexity API to fill in missing player metadata / match details

Outputs Sackmann-format CSV to data/raw/tennis_wta/wta_matches_{year}.csv

Usage:
    python -m src.wta_backfill --years 2025 2026
    python -m src.wta_backfill --check-sackmann     # just check if Sackmann has data
    python -m src.wta_backfill --years 2025 --source firecrawl
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rapidfuzz import process as rfuzz_process, fuzz as rfuzz

from config import RAW_WTA, META_DIR
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
SACKMANN_WTA_BASE = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"
)

SACKMANN_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry", "winner_name",
    "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_seed", "loser_entry", "loser_name",
    "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
]

# Firecrawl API
FIRECRAWL_API_URL = "https://api.firecrawl.dev/v1"

# Tennis Explorer base URL for WTA results
TENNIS_EXPLORER_BASE = "https://www.tennisexplorer.com"

# WTA tournament slugs for Tennis Explorer
# URL pattern: https://www.tennisexplorer.com/{slug}/{year}/wta-women/
WTA_2025_TOURNAMENTS = [
    # Grand Slams
    {"slug": "australian-open", "name": "Australian Open", "surface": "Hard", "draw_size": 128, "level": "G", "tourney_date": "20250113", "tourney_id": "2025-580"},
    {"slug": "french-open", "name": "Roland Garros", "surface": "Clay", "draw_size": 128, "level": "G", "tourney_date": "20250525", "tourney_id": "2025-520"},
    {"slug": "wimbledon", "name": "Wimbledon", "surface": "Grass", "draw_size": 128, "level": "G", "tourney_date": "20250630", "tourney_id": "2025-540"},
    {"slug": "us-open", "name": "US Open", "surface": "Hard", "draw_size": 128, "level": "G", "tourney_date": "20250825", "tourney_id": "2025-560"},
    # WTA 1000
    {"slug": "indian-wells", "name": "Indian Wells", "surface": "Hard", "draw_size": 96, "level": "M", "tourney_date": "20250305", "tourney_id": "2025-404"},
    {"slug": "miami", "name": "Miami", "surface": "Hard", "draw_size": 96, "level": "M", "tourney_date": "20250319", "tourney_id": "2025-403"},
    {"slug": "madrid-wta", "name": "Madrid", "surface": "Clay", "draw_size": 64, "level": "M", "tourney_date": "20250425", "tourney_id": "2025-1536"},
    {"slug": "rome-wta", "name": "Rome", "surface": "Clay", "draw_size": 64, "level": "M", "tourney_date": "20250511", "tourney_id": "2025-416"},
    {"slug": "montreal-wta", "name": "Canada", "surface": "Hard", "draw_size": 64, "level": "M", "tourney_date": "20250809", "tourney_id": "2025-421"},
    {"slug": "cincinnati-wta", "name": "Cincinnati", "surface": "Hard", "draw_size": 64, "level": "M", "tourney_date": "20250816", "tourney_id": "2025-422"},
    {"slug": "beijing", "name": "Beijing", "surface": "Hard", "draw_size": 64, "level": "M", "tourney_date": "20250927", "tourney_id": "2025-747"},
    {"slug": "wuhan", "name": "Wuhan", "surface": "Hard", "draw_size": 64, "level": "M", "tourney_date": "20251019", "tourney_id": "2025-1054"},
    # WTA 500
    {"slug": "brisbane", "name": "Brisbane", "surface": "Hard", "draw_size": 32, "level": "A", "tourney_date": "20250101", "tourney_id": "2025-0339"},
    {"slug": "doha", "name": "Doha", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20250210", "tourney_id": "2025-451"},
    {"slug": "dubai", "name": "Dubai", "surface": "Hard", "draw_size": 32, "level": "A", "tourney_date": "20250217", "tourney_id": "2025-495"},
    {"slug": "berlin", "name": "Berlin", "surface": "Grass", "draw_size": 28, "level": "A", "tourney_date": "20250616", "tourney_id": "2025-2014"},
    {"slug": "eastbourne", "name": "Eastbourne", "surface": "Grass", "draw_size": 28, "level": "A", "tourney_date": "20250623", "tourney_id": "2025-710"},
    {"slug": "tokyo", "name": "Tokyo", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20250920", "tourney_id": "2025-329"},
    {"slug": "seoul-wta", "name": "Seoul", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20250922", "tourney_id": "2025-9326"},
    # WTA 250
    {"slug": "adelaide", "name": "Adelaide", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20250106", "tourney_id": "2025-8998"},
    {"slug": "auckland", "name": "Auckland", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20250106", "tourney_id": "2025-301"},
    {"slug": "abu-dhabi-wta", "name": "Abu Dhabi", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20250203", "tourney_id": "2025-9409"},
    {"slug": "hobart", "name": "Hobart", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20250106", "tourney_id": "2025-352"},
    {"slug": "linz", "name": "Linz", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20250127", "tourney_id": "2025-7694"},
    {"slug": "austin-wta", "name": "Austin", "surface": "Hard", "draw_size": 32, "level": "A", "tourney_date": "20250224", "tourney_id": "2025-2080"},
    {"slug": "charleston", "name": "Charleston", "surface": "Clay", "draw_size": 56, "level": "A", "tourney_date": "20250330", "tourney_id": "2025-804"},
    {"slug": "singapore-wta", "name": "Singapore", "surface": "Hard", "draw_size": 32, "level": "A", "tourney_date": "20250127", "tourney_id": "2025-1052"},
    {"slug": "strasbourg", "name": "Strasbourg", "surface": "Clay", "draw_size": 28, "level": "A", "tourney_date": "20250519", "tourney_id": "2025-337"},
    {"slug": "nottingham", "name": "Nottingham", "surface": "Grass", "draw_size": 28, "level": "A", "tourney_date": "20250609", "tourney_id": "2025-1080"},
    {"slug": "birmingham", "name": "Birmingham", "surface": "Grass", "draw_size": 28, "level": "A", "tourney_date": "20250616", "tourney_id": "2025-742"},
    {"slug": "bad-homburg-wta", "name": "Bad Homburg", "surface": "Grass", "draw_size": 28, "level": "A", "tourney_date": "20250623", "tourney_id": "2025-743"},
]

WTA_2026_TOURNAMENTS = [
    {"slug": "australian-open", "name": "Australian Open", "surface": "Hard", "draw_size": 128, "level": "G", "tourney_date": "20260119", "tourney_id": "2026-580"},
    {"slug": "brisbane", "name": "Brisbane", "surface": "Hard", "draw_size": 32, "level": "A", "tourney_date": "20260101", "tourney_id": "2026-0339"},
    {"slug": "adelaide", "name": "Adelaide", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20260106", "tourney_id": "2026-8998"},
    {"slug": "auckland", "name": "Auckland", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20260106", "tourney_id": "2026-301"},
    {"slug": "hobart", "name": "Hobart", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20260112", "tourney_id": "2026-352"},
    {"slug": "abu-dhabi-wta", "name": "Abu Dhabi", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20260202", "tourney_id": "2026-9409"},
    {"slug": "doha", "name": "Doha", "surface": "Hard", "draw_size": 28, "level": "A", "tourney_date": "20260209", "tourney_id": "2026-451"},
    {"slug": "dubai", "name": "Dubai", "surface": "Hard", "draw_size": 32, "level": "A", "tourney_date": "20260216", "tourney_id": "2026-495"},
    {"slug": "indian-wells", "name": "Indian Wells", "surface": "Hard", "draw_size": 96, "level": "M", "tourney_date": "20260305", "tourney_id": "2026-404"},
]

YEAR_TOURNAMENTS = {
    2025: WTA_2025_TOURNAMENTS,
    2026: WTA_2026_TOURNAMENTS,
}

# ---------------------------------------------------------------------------
# Fast Player Resolution (Issue 1)
# ---------------------------------------------------------------------------
_resolution_cache: dict[str, tuple[str, str|None, int]] = {}

def _build_wta_player_lookup() -> tuple[list[str], dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Build lookup from WTA players only (6,500 vs 65,000)."""
    path = RAW_WTA / "wta_players.csv"
    if not path.exists():
        log.warning(f"WTA players file not found at {path}")
        return [], {}, {}
    
    df = pd.read_csv(path, low_memory=False)
    pool = []
    n2id = {}
    for _, row in df.iterrows():
        try:
            pid = str(int(float(row["player_id"])))
        except (ValueError, TypeError):
            continue
            
        fname = str(row.get("name_first", "")).strip()
        lname = str(row.get("name_last", "")).strip()
        full = f"{fname} {lname}".strip()
        
        norm = _normalize_name(full)
        if norm:
            pool.append(norm)
            n2id[norm] = pid
            
    ln_idx = _build_lastname_index(pool, n2id)
    return pool, n2id, ln_idx

def _build_lastname_index(player_pool: list[str], name_to_id: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """Index: normalized_last_name -> [(full_name, player_id), ...]"""
    idx = {}
    for name, pid in name_to_id.items():
        parts = name.strip().split()
        if parts:
            last = parts[-1].lower()
            idx.setdefault(last, []).append((name, pid))
    return idx

def _resolve_player_fast(name_raw: str, pool: list[str], n2id: dict[str, str], ln_idx: dict[str, list[tuple[str, str]]], overrides: dict[str, str] = None) -> tuple[str, str|None, int]:
    """Fast resolution using rapidfuzz, caching, and lastname index."""
    if not name_raw:
        return "", None, 0
        
    if overrides and name_raw in overrides:
        name_raw = overrides[name_raw]
        
    if name_raw in _resolution_cache:
        return _resolution_cache[name_raw]
        
    norm = _normalize_name(name_raw)
    if not norm:
        return name_raw, None, 0
        
    # Check exact match
    if norm in n2id:
        res = (norm, n2id[norm], 100)
        _resolution_cache[name_raw] = res
        return res
        
    parts = norm.split()
    last = parts[0] if parts else norm
    
    candidates = ln_idx.get(last, [])
    if candidates:
        if len(candidates) == 1:
            res = (candidates[0][0], candidates[0][1], 90)
            _resolution_cache[name_raw] = res
            return res
        else:
            cand_names = [c[0] for c in candidates]
            match = rfuzz_process.extractOne(norm, cand_names, scorer=rfuzz.WRatio)
            if match and match[1] >= 80:
                best_name = match[0]
                pid = n2id[best_name]
                res = (best_name, pid, match[1])
                _resolution_cache[name_raw] = res
                return res

    match = rfuzz_process.extractOne(norm, pool, scorer=rfuzz.WRatio)
    if match and match[1] >= 80:
        res = (match[0], n2id[match[0]], match[1])
        _resolution_cache[name_raw] = res
        return res
        
    return name_raw, None, 0

# ---------------------------------------------------------------------------
# Source 1: Jeff Sackmann GitHub
# ---------------------------------------------------------------------------
def check_sackmann(year: int) -> pd.DataFrame | None:
    """Try downloading WTA match file from Sackmann's GitHub."""
    url = f"{SACKMANN_WTA_BASE}/wta_matches_{year}.csv"
    log.info(f"Checking Sackmann: {url}")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.text) > 100:
            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            log.info(f"  Sackmann HIT: {len(df)} rows for {year}")
            return df
        else:
            log.info(f"  Sackmann MISS: HTTP {resp.status_code}")
            return None
    except Exception as e:
        log.warning(f"  Sackmann error: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 2: Firecrawl + Tennis Explorer
# ---------------------------------------------------------------------------
def _get_firecrawl_key() -> str | None:
    """Get Firecrawl API key from env or firecrawl.txt."""
    key = os.environ.get("FIRECRAWL_API_KEY")
    if key:
        return key
    # Try reading from firecrawl.txt
    for path in [Path("firecrawl.txt"), Path(__file__).parent.parent / "firecrawl.txt"]:
        if path.exists():
            text = path.read_text().strip()
            # Extract the key (second line or after //)
            for line in text.split("\n"):
                line = line.strip()
                if line and not line.startswith("//") and not line.startswith("#"):
                    return line
    return None


def _get_rtrvr_key() -> str | None:
    """Get rtrvr.ai API key from env or rtrvr.ai.txt."""
    key = os.environ.get("RTRVR_API_KEY")
    if key:
        return key
    for path in [Path("rtvt.ai.txt"), Path(__file__).parent.parent / "rtvt.ai.txt"]:
        if path.exists():
            text = path.read_text().strip()
            for line in text.split("\n"):
                line = line.strip()
                if line and not line.startswith("//") and not line.startswith("#"):
                    return line
    return None

def rtrvr_scrape(url: str, api_key: str) -> str | None:
    """Scrape a URL using rtrvr.ai API, returns text content or accessibility tree."""
    try:
        resp = requests.post(
            "https://api.rtrvr.ai/scrape",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"urls": [url]},
            timeout=60,
        )
        if resp.status_code == 401:
            log.warning("  rtrvr.ai unauthorized (401) for %s", url)
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") and data.get("tabs"):
            tab = data["tabs"][0]
            tab_status = str(tab.get("status") or "").lower()
            if tab_status and tab_status != "success":
                log.warning(
                    "  rtrvr.ai tab scrape failed for %s: status=%s error=%s",
                    url,
                    tab.get("status"),
                    tab.get("error"),
                )
                return None
            payload = tab.get("content") or tab.get("tree") or None
            if payload:
                return payload
            log.warning("  rtrvr.ai returned no content/tree for %s", url)
            return None
        log.warning(f"  rtrvr.ai failed: {data.get('error', 'unknown')}")
        return None
    except Exception as e:
        log.warning(f"  rtrvr.ai error: {e}")
        return None


def firecrawl_scrape(url: str, api_key: str) -> dict[str, Any] | None:
    """Scrape a URL using Firecrawl API, returns markdown content."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": url,
        "formats": ["markdown"],
        "waitFor": 3000,  # wait for JS rendering
    }

    try:
        resp = requests.post(
            f"{FIRECRAWL_API_URL}/scrape",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data.get("data", {})
        else:
            log.warning(f"  Firecrawl failed: {data.get('error', 'unknown')}")
            return None
    except Exception as e:
        log.warning(f"  Firecrawl error: {e}")
        return None


def firecrawl_extract(url: str, api_key: str, schema: dict) -> dict[str, Any] | None:
    """Use Firecrawl's extract endpoint to get structured data."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "urls": [url],
        "prompt": (
            "Extract all completed tennis match results from this page. "
            "For each match, extract: round, winner name, loser name, score "
            "(set-by-set like '6-3 7-5'), and match duration in minutes if shown."
        ),
        "schema": schema,
    }

    try:
        resp = requests.post(
            f"{FIRECRAWL_API_URL}/extract",
            headers=headers,
            json=payload,
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data.get("data", {})
        log.warning(f"  Firecrawl extract failed: {data.get('error')}")
        return None
    except Exception as e:
        log.warning(f"  Firecrawl extract error: {e}")
        return None


MATCH_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "round": {"type": "string", "description": "Match round: Final, Semifinal, Quarterfinal, R16, R32, R64, R128, or Round Robin"},
                    "winner_name": {"type": "string", "description": "Full name of the match winner"},
                    "loser_name": {"type": "string", "description": "Full name of the match loser"},
                    "score": {"type": "string", "description": "Match score in set-by-set format like '6-3 7-5' or '7-6(5) 3-6 6-4'"},
                    "minutes": {"type": "integer", "description": "Match duration in minutes, 0 if not shown"},
                    "winner_seed": {"type": "string", "description": "Winner seed number or empty"},
                    "loser_seed": {"type": "string", "description": "Loser seed number or empty"},
                },
                "required": ["winner_name", "loser_name", "score"],
            },
        },
    },
    "required": ["matches"],
}

ROUND_MAP = {
    "final": "F", "finals": "F", "f": "F",
    "semifinal": "SF", "semi-final": "SF", "semi-finals": "SF", "sf": "SF",
    "quarterfinal": "QF", "quarter-final": "QF", "quarter-finals": "QF", "qf": "QF",
    "round of 16": "R16", "r16": "R16", "4th round": "R16", "4r": "R16",
    "round of 32": "R32", "r32": "R32", "3rd round": "R32", "3r": "R32",
    "round of 64": "R64", "r64": "R64", "2nd round": "R64", "2r": "R64",
    "round of 128": "R128", "r128": "R128", "1st round": "R128", "1r": "R128",
    "round robin": "RR", "group": "RR", "rr": "RR",
}


def _round_for_numbered_round(round_number: int, draw_size: int) -> str:
    """Map '1. round', '2. round' etc. to Sackmann round code using draw size."""
    effective_draw = 2 ** (draw_size - 1).bit_length()  # next power of 2
    round_of = effective_draw // (2 ** (round_number - 1))

    if round_of >= 128: return "R128"
    if round_of >= 64: return "R64"
    if round_of >= 32: return "R32"
    if round_of >= 16: return "R16"
    if round_of >= 8: return "QF"
    if round_of >= 4: return "SF"
    return "F"

def _normalize_round(raw: str | None, draw_size: int = 128) -> str:
    """Normalize round string to Sackmann code, taking draw size into account."""
    if not raw:
        return "R32"
        
    val = raw.strip().lower()
    
    # Dynamic round mapping for numbered rounds
    if val in ("1r", "1st round", "1. round", "round of 128", "r128"):
        return _round_for_numbered_round(1, draw_size)
    if val in ("2r", "2nd round", "2. round", "round of 64", "r64"):
        return _round_for_numbered_round(2, draw_size)
    if val in ("3r", "3rd round", "3. round", "round of 32", "r32"):
        return _round_for_numbered_round(3, draw_size)
    if val in ("4r", "4th round", "4. round", "round of 16", "r16"):
        return _round_for_numbered_round(4, draw_size)
        
    return ROUND_MAP.get(val, "R32")


def verify_tournament_urls(year: int) -> list[dict]:
    """Check which tournament URLs return valid data."""
    tournaments = YEAR_TOURNAMENTS.get(year, [])
    results = []
    for t in tournaments:
        url = f"{TENNIS_EXPLORER_BASE}/{t['slug']}/{year}/wta-women/"
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            results.append({
                "name": t["name"],
                "slug": t["slug"],
                "status": resp.status_code,
                "valid": resp.status_code == 200,
            })
        except:
            results.append({"name": t["name"], "slug": t["slug"], "status": 0, "valid": False})
    return results


def scrape_tournament(
    tournament: dict[str, Any],
    year: int,
    firecrawl_key: str | None,
    rtrvr_key: str | None,
    wta_pool: list[str],
    wta_n2id: dict[str, str],
    wta_ln_idx: dict[str, list[tuple[str, str]]],
    full_pool: list[str],
    full_n2id: dict[str, str],
    full_last_idx: dict,
    overrides: dict[str, str],
    player_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scrape a single WTA tournament from Tennis Explorer using Firecrawl / rtrvr.ai.

    Tennis Explorer URL: /slug/year/wta-women/
    Returns rows formatted for the Sackmann-style table structure.
    """
    url = f"{TENNIS_EXPLORER_BASE}/{tournament['slug']}/{year}/wta-women/"
    log.info(f"  Scraping: {url}")

    md = None
    # Try Firecrawl first
    if firecrawl_key:
        scraped = firecrawl_scrape(url, firecrawl_key)
        if scraped and scraped.get("markdown"):
            md = scraped["markdown"]

    # Fallback to rtrvr.ai
    if not md and rtrvr_key:
        log.info("  Firecrawl failed or missing, trying rtrvr.ai...")
        text = rtrvr_scrape(url, rtrvr_key)
        if text:
            md = text  # parser needs to handle both markdown and text

    if not md:
        log.warning(f"  No data from scrapers for {url}")
        return []

    if "404" in md[:200] or len(md) < 300:
        log.warning(f"  Page not found or too short for {tournament['name']}")
        return []

    # Parse either Firecrawl markdown tables or rtrvr accessibility-tree output.
    if "[row]" in md and "[cell]" in md:
        matches_raw = _parse_rtrvr_tree(md, draw_size=tournament.get("draw_size", 128))
    else:
        matches_raw = _parse_tennis_explorer_table(md, draw_size=tournament.get("draw_size", 128))
    log.info(f"  Parsed {len(matches_raw)} matches from Tennis Explorer")

    if not matches_raw:
        return []

    # Convert to Sackmann format
    rows = []
    tourney_id = tournament.get("tourney_id", f"{year}-wta-{tournament['slug']}")

    for i, m in enumerate(matches_raw):
        w_name_raw = m.get("winner_name", "").strip()
        l_name_raw = m.get("loser_name", "").strip()
        score = m.get("score", "").strip()

        if not w_name_raw or not l_name_raw or not score:
            continue

        # Fast Resolve player IDs
        w_name, w_id, _ = _resolve_player_fast(w_name_raw, wta_pool, wta_n2id, wta_ln_idx, overrides)
        if not w_id:
            w_name, w_id, _ = _resolve_player(w_name_raw, full_pool, full_n2id, full_last_idx, overrides)

        l_name, l_id, _ = _resolve_player_fast(l_name_raw, wta_pool, wta_n2id, wta_ln_idx, overrides)
        if not l_id:
            l_name, l_id, _ = _resolve_player(l_name_raw, full_pool, full_n2id, full_last_idx, overrides)

        if not w_id or not l_id:
            log.debug(f"  Skipped: {w_name_raw} vs {l_name_raw} (unresolved)")
            continue

        # Player metadata
        w_meta = player_meta.get(w_id, {})
        l_meta = player_meta.get(l_id, {})

        round_code = _normalize_round(m.get("round"), draw_size=tournament.get("draw_size", 128))

        row = {col: "" for col in SACKMANN_COLUMNS}
        row.update({
            "tourney_id": tourney_id,
            "tourney_name": tournament["name"],
            "surface": tournament["surface"],
            "draw_size": tournament["draw_size"],
            "tourney_level": tournament["level"],
            "tourney_date": tournament["tourney_date"],
            "match_num": i + 1,
            "winner_id": w_id,
            "winner_seed": m.get("winner_seed", ""),
            "winner_name": w_name,
            "winner_hand": w_meta.get("hand", ""),
            "winner_ht": w_meta.get("ht", ""),
            "winner_ioc": w_meta.get("ioc", ""),
            "winner_age": _compute_age(w_meta.get("dob"), tournament["tourney_date"]),
            "loser_id": l_id,
            "loser_seed": m.get("loser_seed", ""),
            "loser_name": l_name,
            "loser_hand": l_meta.get("hand", ""),
            "loser_ht": l_meta.get("ht", ""),
            "loser_ioc": l_meta.get("ioc", ""),
            "loser_age": _compute_age(l_meta.get("dob"), tournament["tourney_date"]),
            "score": score,
            "best_of": 3,
            "round": round_code,
            "minutes": m.get("minutes", ""),
        })
        rows.append(row)

    return rows


def _parse_tennis_explorer_table(md: str, draw_size: int = 128) -> list[dict[str, Any]]:
    """
    Parse Tennis Explorer's markdown table format into match dicts.

    Format can be either a single large table (later rounds) or
    split sections for early rounds (like "1. round", "2. round").
    """
    lines = md.split("\n")
    matches = []
    
    # We maintain a fallback round extracted from text headers like "1. round" or "Quarterfinal"
    fallback_round = "R32"
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect textual round headers
        lower_line = line.lower()
        if "1. round" in lower_line or "1st round" in lower_line or lower_line.startswith("r128"):
            fallback_round = _round_for_numbered_round(1, draw_size)
        elif "2. round" in lower_line or "2nd round" in lower_line or lower_line.startswith("r64"):
            fallback_round = _round_for_numbered_round(2, draw_size)
        elif "3. round" in lower_line or "3rd round" in lower_line or lower_line.startswith("r32"):
            fallback_round = _round_for_numbered_round(3, draw_size)
        elif "4. round" in lower_line or "4th round" in lower_line or "round of 16" in lower_line:
            fallback_round = _round_for_numbered_round(4, draw_size)
        elif "quarterfinal" in lower_line or lower_line.startswith("qf"):
            fallback_round = "QF"
        elif "semifinal" in lower_line or lower_line.startswith("sf"):
            fallback_round = "SF"
        elif "final" in lower_line and "semifinal" not in lower_line and "quarterfinal" not in lower_line:
            fallback_round = "F"
            
        # Parse table rows
        if line.startswith("|"):
            row = line
            cells = [c.strip() for c in row.split("|")]
            cells = [c for c in cells if c or c == ""]

            # Skip header sequences
            if not cells or all(c in ("", "---", "Start", "Round", "S") for c in cells[:5]):
                i += 1
                continue

            # Detect round column strictly in the first few cells
            # Tennis Explorer puts "1R", "2R", "3R", "4R", "QF", "SF", "F" typically in the second cell
            round_cell = ""
            for c in cells[:4]:
                norm = _normalize_round(c, draw_size=draw_size)
                # If the normalized round is not the default R32 OR the cell actually was a valid raw string
                # Note: `_normalize_round` defaults to "R32" if unknown, so we check if the input was explicitly related to a round
                c_lower = c.lower()
                if c_lower in ROUND_MAP or c_lower in ("1r", "2r", "3r", "4r", "f", "sf", "qf", "r16", "r32", "r64", "r128", "rr"):
                    round_cell = norm
                    fallback_round = norm  # Update fallback if the table explicitly defines it
                    break

            # Try to extract player 1 name
            player1_match = re.search(r"\[([^\]]+)\]", row)
            if player1_match:
                player1_name = player1_match.group(1).strip()
                seed1 = ""
                seed_match = re.search(r"\]\([^)]+\)\s*\((\d+)\)", row)
                if seed_match:
                    seed1 = seed_match.group(1)

                set_scores_1 = _extract_set_scores_from_cells(cells)
                sets_won_1 = _extract_sets_won(cells)

                # Fetch player 2 from the NEXT table row
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                    row2 = lines[i + 1].strip()
                    player2_match = re.search(r"\[([^\]]+)\]", row2)
                    
                    if player2_match:
                        player2_name = player2_match.group(1).strip()
                        seed2 = ""
                        seed_match2 = re.search(r"\]\([^)]+\)\s*\((\d+)\)", row2)
                        if seed_match2:
                            seed2 = seed_match2.group(1)

                        cells2 = [c.strip() for c in row2.split("|")]
                        set_scores_2 = _extract_set_scores_from_cells(cells2)
                        sets_won_2 = _extract_sets_won(cells2)

                        score = _build_score_string(set_scores_1, set_scores_2)
                        
                        if score:
                            # Determine winner
                            if sets_won_1 > sets_won_2:
                                winner_name, loser_name = player1_name, player2_name
                                winner_seed, loser_seed = seed1, seed2
                            elif sets_won_2 > sets_won_1:
                                winner_name, loser_name = player2_name, player1_name
                                winner_seed, loser_seed = seed2, seed1
                            else:
                                w1 = sum(1 for s1, s2 in zip(set_scores_1, set_scores_2)
                                          if s1.isdigit() and s2.isdigit() and int(s1) > int(s2))
                                w2 = sum(1 for s1, s2 in zip(set_scores_1, set_scores_2)
                                          if s1.isdigit() and s2.isdigit() and int(s2) > int(s1))
                                if w1 >= w2:
                                    winner_name, loser_name = player1_name, player2_name
                                    winner_seed, loser_seed = seed1, seed2
                                else:
                                    winner_name, loser_name = player2_name, player1_name
                                    winner_seed, loser_seed = seed2, seed1

                            matches.append({
                                "round": round_cell or fallback_round,
                                "winner_name": winner_name,
                                "loser_name": loser_name,
                                "score": score,
                                "winner_seed": winner_seed,
                                "loser_seed": loser_seed,
                            })
                            i += 1  # Skip the next row as we just processed it
                            
        i += 1

    return matches


def _parse_rtrvr_tree(tree_text: str, draw_size: int = 128) -> list[dict[str, Any]]:
    """Parse rtrvr accessibility-tree output into match dicts."""
    def _clean_summary(text: str) -> str:
        clean = text.strip()
        clean = re.sub(r"^\[[^\]]+\]\s*", "", clean).strip()
        clean = re.sub(r"\s+\[[^\]]+\]", "", clean).strip()
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def _split_name_seed(raw_name: str) -> tuple[str, str]:
        text = raw_name.strip()
        seed = ""
        match = re.match(r"^(.*?)(?:\s+\((\d+)\))?$", text)
        if match:
            text = match.group(1).strip()
            seed = match.group(2) or ""
        return text, seed

    def _parse_player_summary(summary: str, *, winner_row: bool) -> tuple[str, str, int | None, list[int]] | None:
        text = _clean_summary(summary)
        if not text:
            return None

        text = re.sub(r"\s+Click for match highlights.*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+info.*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+\d+\.\d+\s+\d+\.\d+\s*$", "", text)

        tokens = text.split()
        set_idx = None
        for idx, token in enumerate(tokens):
            if token in {"0", "1", "2", "3"} and idx > 0:
                set_idx = idx
                break
        if set_idx is None:
            return None

        raw_name = " ".join(tokens[:set_idx]).strip()
        if not raw_name:
            return None

        name, seed = _split_name_seed(raw_name)
        score_tokens = [token for token in tokens[set_idx + 1:] if re.fullmatch(r"\d+", token)]
        set_scores = [int(token) for token in score_tokens]

        try:
            sets_won = int(tokens[set_idx])
        except Exception:
            sets_won = None

        return name, seed, sets_won, set_scores

    def _build_score_from_rows(
        winner_scores: list[int],
        loser_scores: list[int],
        winner_sets: int | None,
        loser_sets: int | None,
    ) -> str:
        if not winner_scores or not loser_scores:
            return ""

        set_count = min(len(winner_scores), len(loser_scores))
        if winner_sets is not None and loser_sets is not None:
            expected_sets = max(winner_sets + loser_sets, 2)
            if expected_sets <= set_count:
                set_count = expected_sets

        base_w = winner_scores[:set_count]
        base_l = loser_scores[:set_count]
        extra_w = winner_scores[set_count:]
        extra_l = loser_scores[set_count:]

        tiebreak_targets = [
            idx
            for idx, (w_val, l_val) in enumerate(zip(base_w, base_l))
            if {w_val, l_val} == {6, 7}
        ]

        score_parts: list[str] = []
        for idx, (w_val, l_val) in enumerate(zip(base_w, base_l)):
            score_text = f"{w_val}-{l_val}"
            if idx in tiebreak_targets:
                if extra_l:
                    score_text += f"({extra_l.pop(0)})"
                elif extra_w:
                    score_text += f"({extra_w.pop(0)})"
            score_parts.append(score_text)

        return " ".join(score_parts)

    row_summaries: list[str] = []
    for raw_line in tree_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("[row]"):
            continue
        summary = _clean_summary(line[len("[row]"):])
        if summary:
            row_summaries.append(summary)

    matches: list[dict[str, Any]] = []
    winner_pattern = re.compile(
        r"^(?P<date>\d{2}\.\d{2}\.)\s+(?P<time>\d{2}:\d{2})\s+(?P<round>F|SF|QF|R16|R32|R64|R128|RR|[1-4]R)\s+(?P<rest>.+)$",
        re.IGNORECASE,
    )

    i = 0
    while i < len(row_summaries):
        winner_match = winner_pattern.match(row_summaries[i])
        if not winner_match:
            i += 1
            continue

        loser_idx = i + 1
        while loser_idx < len(row_summaries) and not row_summaries[loser_idx].strip():
            loser_idx += 1
        if loser_idx >= len(row_summaries):
            break
        if winner_pattern.match(row_summaries[loser_idx]):
            i += 1
            continue

        winner_parsed = _parse_player_summary(winner_match.group("rest"), winner_row=True)
        loser_parsed = _parse_player_summary(row_summaries[loser_idx], winner_row=False)
        if not winner_parsed or not loser_parsed:
            i = loser_idx + 1
            continue

        winner_name, winner_seed, winner_sets, winner_scores = winner_parsed
        loser_name, loser_seed, loser_sets, loser_scores = loser_parsed
        score = _build_score_from_rows(winner_scores, loser_scores, winner_sets, loser_sets)
        if not score:
            i = loser_idx + 1
            continue

        matches.append(
            {
                "round": _normalize_round(winner_match.group("round"), draw_size=draw_size),
                "winner_name": winner_name,
                "loser_name": loser_name,
                "score": score,
                "winner_seed": winner_seed,
                "loser_seed": loser_seed,
            }
        )
        i = loser_idx + 1

    return matches


def _extract_set_scores_from_cells(cells: list[str]) -> list[str]:
    """Extract set scores from table cells. Set scores are single digits 0-7 or '6x' (tiebreak)."""
    scores = []
    for c in cells:
        c = c.strip()
        # Single digit or tiebreak format (like "68" for 6(8))
        if re.match(r"^\d{1,2}$", c) and c not in ("", "0") or c == "0":
            if len(c) <= 2:
                scores.append(c)
        if len(scores) >= 5:
            break
    return scores


def _extract_sets_won(cells: list[str]) -> int:
    """Extract the 'sets won' count (appears right after player name, single digit 0-3)."""
    for c in cells:
        c = c.strip()
        if c in ("0", "1", "2", "3"):
            return int(c)
    return 0


def _build_score_string(scores1: list[str], scores2: list[str]) -> str:
    """Build Sackmann-format score string from individual set scores."""
    n = min(len(scores1), len(scores2))
    if n < 2:  # Need at least the sets_won + 1 set score
        return ""

    # First element might be sets_won, remaining are actual set scores
    # Tennis Explorer format: sets_won | s1 | s2 | s3 | s4 | s5
    # But _extract_set_scores_from_cells may include the sets_won count
    # We need to be smart about this

    # If first score is 0-3 and could be sets_won, try both interpretations
    sets = []
    start = 0
    # Skip the sets_won column (first digit column)
    if n > 2 and scores1[0] in ("0", "1", "2", "3") and scores2[0] in ("0", "1", "2", "3"):
        start = 1

    for idx in range(start, min(n, start + 5)):
        s1 = scores1[idx] if idx < len(scores1) else ""
        s2 = scores2[idx] if idx < len(scores2) else ""
        if s1 and s2:
            # Handle tiebreak scores (e.g., "62" means 6(2))
            if len(s1) == 2 and s1[0] in ("6", "7"):
                s1_main = s1[0]
                tb = s1[1]
                sets.append(f"{s1_main}-{s2}({tb})" if int(s2) >= 6 else f"{s1_main}-{s2}")
            elif len(s2) == 2 and s2[0] in ("6", "7"):
                s2_main = s2[0]
                tb = s2[1]
                sets.append(f"{s1}-{s2_main}({tb})" if int(s1) >= 6 else f"{s1}-{s2_main}")
            else:
                sets.append(f"{s1}-{s2}")

    return " ".join(sets) if sets else ""


# ---------------------------------------------------------------------------
# Source 3: Perplexity API for enrichment
# ---------------------------------------------------------------------------
def perplexity_enrich(
    query: str,
    api_key: str | None = None,
) -> str | None:
    """Query Perplexity API for missing match data."""
    key = api_key or os.environ.get("PERPLEXITY_API_KEY")
    if not key:
        return None

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": "You are a tennis data assistant. Return only structured data in the requested format. No commentary."},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"  Perplexity error: {e}")
        return None


# ---------------------------------------------------------------------------
# Player Metadata
# ---------------------------------------------------------------------------
def _load_player_metadata() -> dict[str, dict[str, Any]]:
    """Load WTA player metadata from Sackmann player CSV."""
    path = RAW_WTA / "wta_players.csv"
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
# Output
# ---------------------------------------------------------------------------
def write_output(
    matches: list[dict[str, Any]],
    year: int,
    incremental: bool = True,
) -> Path:
    """Write matches to Sackmann-format CSV."""
    RAW_WTA.mkdir(parents=True, exist_ok=True)
    output_path = RAW_WTA / f"wta_matches_{year}.csv"

    new_df = pd.DataFrame(matches)
    # Ensure all columns present
    for col in SACKMANN_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[SACKMANN_COLUMNS]

    if incremental and output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path, low_memory=False)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined["_dedup"] = (
            combined["tourney_id"].astype(str) + "|"
            + combined["winner_id"].astype(str) + "|"
            + combined["loser_id"].astype(str) + "|"
            + combined["score"].astype(str)
        )
        combined = combined.drop_duplicates(subset=["_dedup"], keep="last")
        combined = combined.drop(columns=["_dedup"])
        new_df = combined

    new_df = new_df.sort_values(
        by=["tourney_date", "tourney_id", "match_num"],
        ascending=True,
    ).reset_index(drop=True)

    new_df.to_csv(output_path, index=False)
    log.info(f"Written {len(new_df)} rows -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main Waterfall
# ---------------------------------------------------------------------------
def backfill_wta(
    years: list[int] | None = None,
    source: str = "auto",
) -> dict[str, Any]:
    """
    Backfill WTA data using waterfall strategy:
    1. Check Sackmann first
    2. Scrape Tennis Explorer via Firecrawl
    3. Enrich with Perplexity if needed
    """
    if years is None:
        years = [2025, 2026]

    log.info("=" * 60)
    log.info("WTA Data Backfill")
    log.info(f"Years: {years}, Source: {source}")
    log.info("=" * 60)

    # Load player resolution (WTA fast vs full fallback)
    wta_pool, wta_n2id, wta_ln_idx = _build_wta_player_lookup()
    full_pool, full_n2id, full_last_idx = _build_player_lookup()
    overrides = _load_name_overrides()
    player_meta = _load_player_metadata()
    log.info(f"WTA pool: {len(wta_pool)} names, Fallback pool: {len(full_pool)} names, metadata: {len(player_meta)} entries")

    firecrawl_key = _get_firecrawl_key()
    rtrvr_key = _get_rtrvr_key()
    reports: dict[str, Any] = {}

    for year in years:
        log.info(f"\n--- Processing {year} ---")

        # Verify tournament URLs first
        log.info(f"  Verifying {year} tournament URLs...")
        url_results = verify_tournament_urls(year)
        valid_tournaments = [r for r in url_results if r["valid"]]
        log.info(f"  Found {len(valid_tournaments)} valid tournament URLs out of {len(url_results)}")
        for r in url_results:
            if not r["valid"]:
                log.warning(f"  URL verification failed for: {r['name']} ({r['slug']}) - Status {r['status']}")

        # Source 1: Sackmann
        if source in ("auto", "sackmann"):
            df = check_sackmann(year)
            if df is not None and len(df) > 50:
                output_path = RAW_WTA / f"wta_matches_{year}.csv"
                RAW_WTA.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)
                reports[str(year)] = {
                    "success": True,
                    "source": "sackmann",
                    "rows": len(df),
                    "output": str(output_path),
                }
                log.info(f"  Sackmann data used: {len(df)} rows")
                continue

        # Source 2: Firecrawl / rtrvr.ai + Tennis Explorer
        if source in ("auto", "firecrawl") and (firecrawl_key or rtrvr_key):
            log.info("  Trying Firecrawl / rtrvr.ai + Tennis Explorer...")
            all_matches = []
            tournaments = YEAR_TOURNAMENTS.get(year, [])
            if not tournaments:
                log.warning(f"  No tournament registry for {year}")

            today_str = datetime.now(UTC).strftime("%Y%m%d")
            for i, t in enumerate(tournaments):
                log.info(f"  [{i+1}/{len(tournaments)}] {t['name']}")
                
                if t.get("tourney_date", "") > today_str:
                    log.info(f"  Skipping future tournament: {t['name']} ({t['tourney_date']})")
                    continue
                
                # Check if URL was verified as valid
                if not any(v["slug"] == t["slug"] for v in valid_tournaments):
                    log.warning(f"  Skipping {t['name']} due to invalid URL")
                    continue
                    
                try:
                    matches = scrape_tournament(
                        tournament=t,
                        year=year,
                        firecrawl_key=firecrawl_key,
                        rtrvr_key=rtrvr_key,
                        wta_pool=wta_pool,
                        wta_n2id=wta_n2id,
                        wta_ln_idx=wta_ln_idx,
                        full_pool=full_pool,
                        full_n2id=full_n2id,
                        full_last_idx=full_last_idx,
                        overrides=overrides,
                        player_meta=player_meta,
                    )
                    all_matches.extend(matches)
                    # Rate limit between requests
                    time.sleep(2)
                except Exception as e:
                    log.error(f"  Error scraping {t['name']}: {e}")
                    continue

            if all_matches:
                output_path = write_output(all_matches, year)
                reports[str(year)] = {
                    "success": True,
                    "source": "firecrawl",
                    "rows": len(all_matches),
                    "output": str(output_path),
                }
                continue

        # Source 3: Perplexity fallback (for individual missing matches)
        log.warning(f"  No automated source worked for WTA {year}")
        reports[str(year)] = {
            "success": False,
            "source": "none",
            "message": "No automated WTA source available. Use gemini_data_prompt.md for manual data.",
        }

    log.info(f"\n{'=' * 60}")
    log.info("WTA BACKFILL COMPLETE")
    for yr, rep in reports.items():
        log.info(f"  {yr}: {rep.get('source', '?')} - {rep.get('rows', 0)} rows")
    log.info(f"{'=' * 60}")

    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WTA match data backfill")
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    parser.add_argument("--source", choices=["auto", "sackmann", "firecrawl"], default="auto")
    parser.add_argument("--check-sackmann", action="store_true",
                        help="Only check if Sackmann has data, don't scrape")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.check_sackmann:
        for year in args.years:
            df = check_sackmann(year)
            if df is not None:
                print(f"{year}: {len(df)} rows available")
            else:
                print(f"{year}: NOT AVAILABLE")
        return

    report = backfill_wta(years=args.years, source=args.source)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
