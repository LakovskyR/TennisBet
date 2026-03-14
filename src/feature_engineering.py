from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    DATE_FMT,
    LAST_UPDATE_FILE,
    META_DIR,
    PROCESSED_DIR,
    RAW_ATP,
    RAW_WTA,
)

DEFAULT_WIN_PCT = 0.5
DEFAULT_HOME_WIN_PCT = 0.5
DEFAULT_ACE_PCT = 0.06
DEFAULT_FIRST_SERVE_PCT = 0.62
DEFAULT_BP_SAVE_PCT = 0.60
DEFAULT_DAYS_SINCE_LAST = 14

LEVEL_MAP = {
    "G": "Grand Slam",
    "M": "Masters",
    "A": "Tour",
    "C": "Challenger",
    "D": "Other",
    "F": "Finals",
    "I": "Team Event",
}

TOUR_FILES = {
    "atp": {
        "matches": PROCESSED_DIR / "atp_matches_master.csv",
        "elo": PROCESSED_DIR / "atp_elo_ratings.csv",
        "players": RAW_ATP / "atp_players.csv",
        "output": PROCESSED_DIR / "atp_player_features.csv",
    },
    "wta": {
        "matches": PROCESSED_DIR / "wta_matches_master.csv",
        "elo": PROCESSED_DIR / "wta_elo_ratings.csv",
        "players": RAW_WTA / "wta_players.csv",
        "output": PROCESSED_DIR / "wta_player_features.csv",
    },
}


def _safe_json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    return json.loads(text)


def _safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_name(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    clean = name.lower().strip()
    for token in [".", "'", "-", "_", "/"]:
        clean = clean.replace(token, " ")
    clean = " ".join(clean.split())
    return clean


def _as_player_id(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return str(int(float(text)))
    except Exception:
        return text


def _to_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _rolling_win_pct(results: deque[int], window: int) -> float:
    if not results:
        return DEFAULT_WIN_PCT
    values = list(results)[-window:]
    if not values:
        return DEFAULT_WIN_PCT
    return float(sum(values) / len(values))


def _rolling_streak5(results: deque[int]) -> int:
    if not results:
        return 0
    values = list(results)[-5:]
    return int(sum(1 if v == 1 else -1 for v in values))


def _series_mean(values: deque[float], default: float) -> float:
    if not values:
        return default
    arr = [v for v in values if v is not None and not math.isnan(v)]
    if not arr:
        return default
    return float(np.mean(arr))


def _new_player_state() -> dict[str, Any]:
    return {
        "results": deque(maxlen=50),
        "surface_results": defaultdict(lambda: deque(maxlen=20)),
        "current_win_streak": 0,
        "current_lose_streak": 0,
        "matches_played": 0,
        "last_match_date": None,
        "match_dates": deque(),
        "sets_last_window": deque(),
        "ace_pct": deque(maxlen=20),
        "first_serve_pct": deque(maxlen=20),
        "bp_save_pct": deque(maxlen=20),
        "home_wins": 0,
        "home_matches": 0,
        "title_dates": deque(),
    }


def _load_tournament_country_map(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    raw = _safe_json_load(path)
    exact = {
        str(k): str(v).upper()
        for k, v in raw.items()
        if isinstance(k, str) and isinstance(v, str) and len(v.strip()) == 3
    }
    normalized = {_normalize_name(k): v for k, v in exact.items()}
    return exact, normalized


def _resolve_tournament_country(
    tourney_name: str | None,
    map_exact: dict[str, str],
    map_norm: dict[str, str],
) -> str | None:
    if not isinstance(tourney_name, str) or not tourney_name.strip():
        return None
    if tourney_name in map_exact:
        return map_exact[tourney_name]
    return map_norm.get(_normalize_name(tourney_name))


def _load_player_ioc_map(players_file: Path) -> dict[str, str]:
    if not players_file.exists():
        return {}
    try:
        players = pd.read_csv(players_file, usecols=["player_id", "ioc"], low_memory=False)
    except Exception:
        return {}

    mapping: dict[str, str] = {}
    for pid, ioc in players.itertuples(index=False):
        player_id = _as_player_id(pid)
        if player_id and isinstance(ioc, str) and ioc.strip():
            mapping[player_id] = ioc.strip().upper()
    return mapping


def _load_elo_pre_map(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    if not path.exists():
        return {}

    usecols = ["match_key", "player_id", "elo_pre", "surface_elo_pre"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    mapping: dict[tuple[str, str], tuple[float, float]] = {}
    for match_key, player_id, elo_pre, surface_pre in df.itertuples(index=False):
        pid = _as_player_id(player_id)
        if not pid:
            continue
        key = (str(match_key), pid)
        mapping[key] = (
            float(elo_pre) if pd.notna(elo_pre) else 1500.0,
            float(surface_pre) if pd.notna(surface_pre) else 1500.0,
        )
    return mapping


def _prepare_matches(path: Path) -> pd.DataFrame:
    keep_cols = [
        "match_key",
        "match_date",
        "tourney_id",
        "tourney_name",
        "surface",
        "tourney_level",
        "tournament_level",
        "round",
        "best_of",
        "match_num",
        "winner_id",
        "winner_name",
        "winner_ioc",
        "winner_rank",
        "loser_id",
        "loser_name",
        "loser_ioc",
        "loser_rank",
        "winner_sets_won",
        "loser_sets_won",
        "w_ace",
        "w_svpt",
        "w_1stIn",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_svpt",
        "l_1stIn",
        "l_bpSaved",
        "l_bpFaced",
        "is_training_eligible",
    ]

    df = pd.read_csv(path, usecols=lambda c: c in keep_cols, low_memory=False)

    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df[df["match_date"].notna()].copy()

    elig = df["is_training_eligible"].astype("string").str.lower()
    if elig.isin(["true", "1"]).any():
        df = df[elig.isin(["true", "1"])].copy()

    df = df.sort_values(["match_date", "tourney_id", "match_num"], na_position="last")
    return df


def _compute_metric_ratios(
    ace: Any,
    svpt: Any,
    first_in: Any,
    bp_saved: Any,
    bp_faced: Any,
) -> tuple[float | None, float | None, float | None]:
    ace_pct = None
    first_serve_pct = None
    bp_save_pct = None

    svpt_v = _to_float(svpt)
    if svpt_v and svpt_v > 0:
        ace_v = _to_float(ace)
        first_in_v = _to_float(first_in)
        if ace_v is not None:
            ace_pct = ace_v / svpt_v
        if first_in_v is not None:
            first_serve_pct = first_in_v / svpt_v

    bp_faced_v = _to_float(bp_faced)
    bp_saved_v = _to_float(bp_saved)
    if bp_faced_v and bp_faced_v > 0 and bp_saved_v is not None:
        bp_save_pct = bp_saved_v / bp_faced_v

    return ace_pct, first_serve_pct, bp_save_pct


def _build_features_for_tour(
    tour: str,
    tournament_country_path: Path,
) -> dict[str, Any]:
    files = TOUR_FILES[tour]
    matches = _prepare_matches(files["matches"])
    elo_map = _load_elo_pre_map(files["elo"])

    tournament_country_exact, tournament_country_norm = _load_tournament_country_map(tournament_country_path)
    player_ioc = _load_player_ioc_map(files["players"])

    player_state: dict[str, dict[str, Any]] = defaultdict(_new_player_state)
    h2h_state: dict[tuple[str, str], dict[str, Any]] = {}
    tournament_wins: dict[tuple[str, str], int] = defaultdict(int)

    rows: list[dict[str, Any]] = []

    known_ioc_players = 0
    unknown_ioc_players = 0

    cols = [
        "match_key",
        "match_date",
        "tourney_id",
        "tourney_name",
        "surface",
        "tourney_level",
        "tournament_level",
        "round",
        "best_of",
        "winner_id",
        "winner_name",
        "winner_ioc",
        "winner_rank",
        "loser_id",
        "loser_name",
        "loser_ioc",
        "loser_rank",
        "winner_sets_won",
        "loser_sets_won",
        "w_ace",
        "w_svpt",
        "w_1stIn",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_svpt",
        "l_1stIn",
        "l_bpSaved",
        "l_bpFaced",
    ]

    for record in matches[cols].itertuples(index=False, name=None):
        (
            match_key,
            match_date,
            tourney_id,
            tourney_name,
            surface,
            tourney_level,
            tournament_level,
            round_name,
            best_of,
            winner_id_raw,
            winner_name,
            winner_ioc_row,
            winner_rank,
            loser_id_raw,
            loser_name,
            loser_ioc_row,
            loser_rank,
            winner_sets,
            loser_sets,
            w_ace,
            w_svpt,
            w_1st_in,
            w_bp_saved,
            w_bp_faced,
            l_ace,
            l_svpt,
            l_1st_in,
            l_bp_saved,
            l_bp_faced,
        ) = record

        winner_id = _as_player_id(winner_id_raw)
        loser_id = _as_player_id(loser_id_raw)
        if not winner_id or not loser_id:
            continue
        if winner_id == loser_id:
            # Corrupt rows occasionally appear with placeholder self-opponents.
            continue

        if isinstance(winner_ioc_row, str) and winner_ioc_row.strip():
            player_ioc[winner_id] = winner_ioc_row.strip().upper()
        if isinstance(loser_ioc_row, str) and loser_ioc_row.strip():
            player_ioc[loser_id] = loser_ioc_row.strip().upper()

        w_rank = _to_float(winner_rank)
        l_rank = _to_float(loser_rank)

        if w_rank is not None and l_rank is not None:
            p1_is_winner = w_rank <= l_rank
        else:
            p1_is_winner = True

        if p1_is_winner:
            p1_id, p2_id = winner_id, loser_id
            p1_name, p2_name = winner_name, loser_name
            p1_rank, p2_rank = w_rank, l_rank
            p1_result = 1
        else:
            p1_id, p2_id = loser_id, winner_id
            p1_name, p2_name = loser_name, winner_name
            p1_rank, p2_rank = l_rank, w_rank
            p1_result = 0

        surface_name = str(surface) if isinstance(surface, str) and surface else "Unknown"
        round_value = str(round_name) if pd.notna(round_name) else "Unknown"

        level_code = str(tourney_level) if pd.notna(tourney_level) else ""
        level_value = LEVEL_MAP.get(level_code, str(tournament_level) if pd.notna(tournament_level) else "Other")

        best_of_value = str(int(best_of)) if pd.notna(best_of) else "3"

        p1_state = player_state[p1_id]
        p2_state = player_state[p2_id]

        p1_elo = elo_map.get((str(match_key), p1_id), (1500.0, 1500.0))
        p2_elo = elo_map.get((str(match_key), p2_id), (1500.0, 1500.0))

        p1_surface_results = p1_state["surface_results"][surface_name]
        p2_surface_results = p2_state["surface_results"][surface_name]

        h2h_key = tuple(sorted([p1_id, p2_id]))
        if h2h_key not in h2h_state:
            h2h_state[h2h_key] = {
                "total": 0,
                "wins": defaultdict(int),
                "surface": defaultdict(lambda: defaultdict(int)),
            }
        h2h = h2h_state[h2h_key]

        tournament_country = _resolve_tournament_country(
            tourney_name,
            tournament_country_exact,
            tournament_country_norm,
        )

        p1_ioc = player_ioc.get(p1_id)
        p2_ioc = player_ioc.get(p2_id)
        known_ioc_players += int(p1_ioc is not None) + int(p2_ioc is not None)
        unknown_ioc_players += int(p1_ioc is None) + int(p2_ioc is None)

        p1_is_home = int(p1_ioc is not None and tournament_country is not None and p1_ioc == tournament_country)
        p2_is_home = int(p2_ioc is not None and tournament_country is not None and p2_ioc == tournament_country)

        if p1_is_home == 1 and p2_is_home == 0:
            home_advantage_flag = 1
        elif p2_is_home == 1 and p1_is_home == 0:
            home_advantage_flag = -1
        else:
            home_advantage_flag = 0

        p1_home_win_pct = (
            p1_state["home_wins"] / p1_state["home_matches"]
            if p1_state["home_matches"] >= 5
            else DEFAULT_HOME_WIN_PCT
        )
        p2_home_win_pct = (
            p2_state["home_wins"] / p2_state["home_matches"]
            if p2_state["home_matches"] >= 5
            else DEFAULT_HOME_WIN_PCT
        )

        p1_last_date = p1_state["last_match_date"]
        p2_last_date = p2_state["last_match_date"]
        p1_days_since = int((match_date - p1_last_date).days) if p1_last_date is not None else DEFAULT_DAYS_SINCE_LAST
        p2_days_since = int((match_date - p2_last_date).days) if p2_last_date is not None else DEFAULT_DAYS_SINCE_LAST

        for st in (p1_state, p2_state):
            cutoff_14 = match_date - pd.Timedelta(days=14)
            while st["match_dates"] and st["match_dates"][0] < cutoff_14:
                st["match_dates"].popleft()

            cutoff_7 = match_date - pd.Timedelta(days=7)
            while st["sets_last_window"] and st["sets_last_window"][0][0] < cutoff_7:
                st["sets_last_window"].popleft()

            cutoff_365 = match_date - pd.Timedelta(days=365)
            while st["title_dates"] and st["title_dates"][0] < cutoff_365:
                st["title_dates"].popleft()

        p1_matches_last_14d = len(p1_state["match_dates"])
        p2_matches_last_14d = len(p2_state["match_dates"])
        p1_sets_last_7d = int(sum(v for _, v in p1_state["sets_last_window"]))
        p2_sets_last_7d = int(sum(v for _, v in p2_state["sets_last_window"]))

        p1_titles_12m = int(len(p1_state["title_dates"]))
        p2_titles_12m = int(len(p2_state["title_dates"]))

        row = {
            "match_key": str(match_key),
            "match_date": match_date.strftime(DATE_FMT),
            "tour": tour,
            "p1_id": p1_id,
            "p2_id": p2_id,
            "p1_name": p1_name,
            "p2_name": p2_name,
            "p1_rank": p1_rank if p1_rank is not None else np.nan,
            "p2_rank": p2_rank if p2_rank is not None else np.nan,
            "p1_elo_overall": p1_elo[0],
            "p2_elo_overall": p2_elo[0],
            "p1_elo_surface": p1_elo[1],
            "p2_elo_surface": p2_elo[1],
            "elo_diff_overall": p1_elo[0] - p2_elo[0],
            "elo_diff_surface": p1_elo[1] - p2_elo[1],
            "p1_win_pct_5": _rolling_win_pct(p1_state["results"], 5),
            "p1_win_pct_10": _rolling_win_pct(p1_state["results"], 10),
            "p1_win_pct_20": _rolling_win_pct(p1_state["results"], 20),
            "p2_win_pct_5": _rolling_win_pct(p2_state["results"], 5),
            "p2_win_pct_10": _rolling_win_pct(p2_state["results"], 10),
            "p2_win_pct_20": _rolling_win_pct(p2_state["results"], 20),
            "p1_win_pct_surface_10": _rolling_win_pct(p1_surface_results, 10),
            "p2_win_pct_surface_10": _rolling_win_pct(p2_surface_results, 10),
            "h2h_p1_wins": int(h2h["wins"][p1_id]),
            "h2h_p2_wins": int(h2h["wins"][p2_id]),
            "h2h_total": int(h2h["total"]),
            "h2h_p1_win_pct": (
                (h2h["wins"][p1_id] / h2h["total"]) if h2h["total"] > 0 else DEFAULT_WIN_PCT
            ),
            "h2h_surface_p1_wins": int(h2h["surface"][surface_name][p1_id]),
            "h2h_surface_p2_wins": int(h2h["surface"][surface_name][p2_id]),
            "p1_current_win_streak": int(p1_state["current_win_streak"]),
            "p2_current_win_streak": int(p2_state["current_win_streak"]),
            "p1_current_lose_streak": int(p1_state["current_lose_streak"]),
            "p2_current_lose_streak": int(p2_state["current_lose_streak"]),
            "p1_streak_5": _rolling_streak5(p1_state["results"]),
            "p2_streak_5": _rolling_streak5(p2_state["results"]),
            "p1_tournament_wins_current": int(tournament_wins[(str(tourney_id), p1_id)]),
            "p2_tournament_wins_current": int(tournament_wins[(str(tourney_id), p2_id)]),
            "p1_title_count_12m": p1_titles_12m,
            "p2_title_count_12m": p2_titles_12m,
            "p1_is_home": p1_is_home,
            "p2_is_home": p2_is_home,
            "home_advantage_flag": home_advantage_flag,
            "p1_home_win_pct": p1_home_win_pct,
            "p2_home_win_pct": p2_home_win_pct,
            "p1_days_since_last_match": p1_days_since,
            "p2_days_since_last_match": p2_days_since,
            "p1_matches_last_14d": p1_matches_last_14d,
            "p2_matches_last_14d": p2_matches_last_14d,
            "p1_sets_played_last_7d": p1_sets_last_7d,
            "p2_sets_played_last_7d": p2_sets_last_7d,
            "p1_ace_pct": _series_mean(p1_state["ace_pct"], DEFAULT_ACE_PCT),
            "p2_ace_pct": _series_mean(p2_state["ace_pct"], DEFAULT_ACE_PCT),
            "p1_1st_serve_pct": _series_mean(p1_state["first_serve_pct"], DEFAULT_FIRST_SERVE_PCT),
            "p2_1st_serve_pct": _series_mean(p2_state["first_serve_pct"], DEFAULT_FIRST_SERVE_PCT),
            "p1_bp_save_pct": _series_mean(p1_state["bp_save_pct"], DEFAULT_BP_SAVE_PCT),
            "p2_bp_save_pct": _series_mean(p2_state["bp_save_pct"], DEFAULT_BP_SAVE_PCT),
            "surface": surface_name,
            "tournament_level": level_value,
            "round": round_value,
            "best_of": best_of_value,
            "p1_wins": p1_result,
            "p1_matches_played_before": int(p1_state["matches_played"]),
            "p2_matches_played_before": int(p2_state["matches_played"]),
        }
        rows.append(row)

        winner_sets_v = _to_float(winner_sets)
        loser_sets_v = _to_float(loser_sets)
        if winner_sets_v is None or loser_sets_v is None:
            sets_total = 2 if best_of_value == "3" else 3
        else:
            sets_total = int(winner_sets_v + loser_sets_v)
            if sets_total <= 0:
                sets_total = 2 if best_of_value == "3" else 3

        winner_serve = _compute_metric_ratios(w_ace, w_svpt, w_1st_in, w_bp_saved, w_bp_faced)
        loser_serve = _compute_metric_ratios(l_ace, l_svpt, l_1st_in, l_bp_saved, l_bp_faced)

        winner_home = int(player_ioc.get(winner_id) is not None and tournament_country is not None and player_ioc[winner_id] == tournament_country)
        loser_home = int(player_ioc.get(loser_id) is not None and tournament_country is not None and player_ioc[loser_id] == tournament_country)

        for pid, result, serve_metrics, is_home in [
            (winner_id, 1, winner_serve, winner_home),
            (loser_id, 0, loser_serve, loser_home),
        ]:
            st = player_state[pid]
            st["matches_played"] += 1
            st["results"].append(result)
            st["surface_results"][surface_name].append(result)

            if result == 1:
                st["current_win_streak"] += 1
                st["current_lose_streak"] = 0
            else:
                st["current_lose_streak"] += 1
                st["current_win_streak"] = 0

            st["last_match_date"] = match_date
            st["match_dates"].append(match_date)
            st["sets_last_window"].append((match_date, sets_total))

            ace_pct, first_serve_pct, bp_save_pct = serve_metrics
            if ace_pct is not None:
                st["ace_pct"].append(float(ace_pct))
            if first_serve_pct is not None:
                st["first_serve_pct"].append(float(first_serve_pct))
            if bp_save_pct is not None:
                st["bp_save_pct"].append(float(bp_save_pct))

            if is_home:
                st["home_matches"] += 1
                if result == 1:
                    st["home_wins"] += 1

        h2h["total"] += 1
        h2h["wins"][winner_id] += 1
        h2h["surface"][surface_name][winner_id] += 1

        tournament_wins[(str(tourney_id), winner_id)] += 1

        if round_value == "F":
            player_state[winner_id]["title_dates"].append(match_date)

    features = pd.DataFrame(rows)
    features = features.sort_values(["match_date", "match_key"], na_position="last")

    output_file = files["output"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_file, index=False)

    win_pct_cols = [
        "p1_win_pct_5",
        "p1_win_pct_10",
        "p1_win_pct_20",
        "p2_win_pct_5",
        "p2_win_pct_10",
        "p2_win_pct_20",
        "p1_home_win_pct",
        "p2_home_win_pct",
    ]

    win_pct_bounds = (
        features[win_pct_cols]
        .ge(0)
        .where(features[win_pct_cols].notna(), True)
        .all()
        .all()
        and features[win_pct_cols]
        .le(1)
        .where(features[win_pct_cols].notna(), True)
        .all()
        .all()
    )

    checks = {
        "rows": int(len(features)),
        "days_since_non_negative": bool(
            (features["p1_days_since_last_match"] >= 0).all() and (features["p2_days_since_last_match"] >= 0).all()
        ),
        "win_pct_bounds": bool(win_pct_bounds),
        "h2h_consistent": bool((features["h2h_total"] >= features["h2h_p1_wins"] + features["h2h_p2_wins"]).all()),
    }

    sample_cols = [
        "match_date",
        "match_key",
        "p1_id",
        "p2_id",
        "p1_days_since_last_match",
        "p2_days_since_last_match",
        "h2h_total",
        "p1_matches_played_before",
        "p2_matches_played_before",
        "p1_wins",
    ]
    sample_rows = features[sample_cols].sample(n=min(10, len(features)), random_state=42).to_dict(orient="records")

    audit_payload = {
        "tour": tour,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "sample": sample_rows,
    }
    _safe_json_dump(META_DIR / f"leakage_audit_{tour}.json", audit_payload)

    total_ioc_obs = known_ioc_players + unknown_ioc_players
    ioc_coverage = (known_ioc_players / total_ioc_obs) if total_ioc_obs else 0.0

    return {
        "tour": tour,
        "output": str(output_file),
        "rows": int(len(features)),
        "ioc_coverage": ioc_coverage,
        "checks": checks,
    }


def build_features(tours: tuple[str, ...] = ("atp", "wta")) -> dict[str, Any]:
    tournament_country_path = Path("tournament_country.json")
    results = {tour: _build_features_for_tour(tour, tournament_country_path) for tour in tours}

    tables = []
    for tour in tours:
        path = TOUR_FILES[tour]["output"]
        tables.append(pd.read_csv(path, low_memory=False))

    combined = pd.concat(tables, ignore_index=True)
    combined = combined.sort_values(["match_date", "tour", "match_key"], na_position="last")
    combined_output = PROCESSED_DIR / "player_features.csv"
    combined.to_csv(combined_output, index=False)

    state = _safe_json_load(LAST_UPDATE_FILE)
    state.update(
        {
            "feature_last_run": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feature_rows": int(len(combined)),
            "feature_outputs": {
                **results,
                "combined": str(combined_output),
            },
        }
    )
    _safe_json_dump(LAST_UPDATE_FILE, state)

    return {
        **results,
        "combined_rows": int(len(combined)),
        "combined_output": str(combined_output),
    }


def main() -> None:
    results = build_features()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
