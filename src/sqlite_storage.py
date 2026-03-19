from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    BANKROLL_LOG_FILE,
    META_DIR,
    ODDS_HISTORY_FILE,
    ODDS_UPCOMING_FILE,
    PROCESSED_DIR,
    RAW_ATP,
    RAW_WTA,
    SQLITE_DB_FILE,
)
from src.player_aliases import load_player_aliases, normalize_player_name


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS players (
        player_id TEXT PRIMARY KEY,
        canonical_name TEXT NOT NULL,
        normalized_name TEXT,
        tour TEXT,
        ioc TEXT,
        source TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_aliases (
        alias TEXT PRIMARY KEY,
        normalized_alias TEXT NOT NULL,
        canonical_name TEXT NOT NULL,
        normalized_canonical_name TEXT NOT NULL,
        player_id TEXT,
        source TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tournaments (
        tournament_key TEXT PRIMARY KEY,
        tour TEXT,
        tournament_name TEXT NOT NULL,
        normalized_name TEXT,
        surface TEXT,
        tournament_level TEXT,
        source TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS matches (
        match_key TEXT PRIMARY KEY,
        tour TEXT NOT NULL,
        match_date TEXT,
        tournament_key TEXT,
        tourney_id TEXT,
        tourney_name TEXT,
        surface TEXT,
        draw_size INTEGER,
        tournament_level TEXT,
        source TEXT,
        source_file TEXT,
        tourney_date TEXT,
        match_num TEXT,
        winner_id TEXT,
        winner_name TEXT,
        loser_id TEXT,
        loser_name TEXT,
        score TEXT,
        best_of TEXT,
        round TEXT,
        winner_sets_won INTEGER,
        loser_sets_won INTEGER,
        total_games INTEGER,
        is_retirement INTEGER,
        is_walkover INTEGER,
        is_training_eligible INTEGER,
        year INTEGER,
        days_since_epoch INTEGER,
        raw_payload_json TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elo_ratings (
        rating_key TEXT PRIMARY KEY,
        tour TEXT NOT NULL,
        match_key TEXT NOT NULL,
        match_date TEXT,
        player_id TEXT,
        player_name TEXT,
        opponent_id TEXT,
        opponent_name TEXT,
        result INTEGER,
        surface TEXT,
        tourney_id TEXT,
        tourney_name TEXT,
        tourney_level TEXT,
        round TEXT,
        k_factor REAL,
        elo_pre REAL,
        elo_post REAL,
        surface_elo_pre REAL,
        surface_elo_post REAL,
        raw_payload_json TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elo_snapshot (
        snapshot_key TEXT PRIMARY KEY,
        tour TEXT NOT NULL,
        player_id TEXT NOT NULL,
        surface TEXT NOT NULL,
        elo REAL NOT NULL,
        raw_payload_json TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_features (
        feature_key TEXT PRIMARY KEY,
        tour TEXT NOT NULL,
        match_key TEXT NOT NULL,
        match_date TEXT,
        p1_id TEXT,
        p2_id TEXT,
        p1_name TEXT,
        p2_name TEXT,
        surface TEXT,
        tournament_level TEXT,
        round TEXT,
        best_of TEXT,
        p1_wins INTEGER,
        raw_payload_json TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS odds_snapshots (
        snapshot_key TEXT PRIMARY KEY,
        match_id TEXT,
        match_date TEXT,
        match_time TEXT,
        tour TEXT,
        tournament TEXT,
        surface TEXT,
        player_1 TEXT,
        player_2 TEXT,
        player_1_resolved TEXT,
        player_2_resolved TEXT,
        player_1_id TEXT,
        player_2_id TEXT,
        player_1_match_score REAL,
        player_2_match_score REAL,
        odds_p1 REAL,
        odds_p2 REAL,
        bookmaker TEXT,
        bookmaker_count INTEGER,
        aggregation_method TEXT,
        source_url TEXT,
        captured_at TEXT,
        is_current_upcoming INTEGER NOT NULL DEFAULT 0,
        raw_payload_json TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id TEXT PRIMARY KEY,
        created_at TEXT,
        match_date TEXT,
        tour TEXT,
        match TEXT,
        p1_name TEXT,
        p2_name TEXT,
        bet_side TEXT,
        probability REAL,
        odds REAL,
        edge REAL,
        stake REAL,
        confidence_tier TEXT,
        model_agreement REAL,
        result TEXT,
        pnl REAL,
        status TEXT,
        resolved_at TEXT,
        raw_payload_json TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bankroll_state (
        state_key TEXT PRIMARY KEY,
        capital REAL NOT NULL,
        updated_at TEXT,
        history_count INTEGER NOT NULL,
        raw_payload_json TEXT,
        synced_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bankroll_events (
        event_id TEXT PRIMARY KEY,
        timestamp TEXT,
        prediction_id TEXT,
        note TEXT,
        pnl REAL,
        capital_before REAL,
        capital_after REAL,
        raw_payload_json TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_runs (
        source_name TEXT PRIMARY KEY,
        last_synced_at TEXT NOT NULL,
        rows_synced INTEGER NOT NULL DEFAULT 0,
        detail_json TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_matches_match_date ON matches(match_date)",
    "CREATE INDEX IF NOT EXISTS idx_matches_tournament_key ON matches(tournament_key)",
    "CREATE INDEX IF NOT EXISTS idx_elo_ratings_match_key ON elo_ratings(match_key)",
    "CREATE INDEX IF NOT EXISTS idx_elo_ratings_player_id ON elo_ratings(player_id)",
    "CREATE INDEX IF NOT EXISTS idx_player_features_match_date ON player_features(match_date)",
    "CREATE INDEX IF NOT EXISTS idx_player_features_match_key ON player_features(match_key)",
    "CREATE INDEX IF NOT EXISTS idx_odds_match_id_captured ON odds_snapshots(match_id, captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_match_date ON predictions(match_date)",
]
SCHEMA_COLUMN_MIGRATIONS = {
    "odds_snapshots": {
        "bookmaker": "TEXT",
        "bookmaker_count": "INTEGER",
        "aggregation_method": "TEXT",
    },
}
ODDS_FRAME_COLUMNS = [
    "match_date",
    "match_time",
    "tour",
    "tournament",
    "surface",
    "player_1",
    "player_2",
    "player_1_resolved",
    "player_2_resolved",
    "player_1_id",
    "player_2_id",
    "player_1_match_score",
    "player_2_match_score",
    "odds_p1",
    "odds_p2",
    "bookmaker",
    "bookmaker_count",
    "aggregation_method",
    "source_url",
    "match_id",
    "captured_at",
]
PREDICTION_FRAME_COLUMNS = [
    "prediction_id",
    "created_at",
    "match_date",
    "tour",
    "match",
    "p1_name",
    "p2_name",
    "bet_side",
    "probability",
    "odds",
    "edge",
    "stake",
    "confidence_tier",
    "model_agreement",
    "result",
    "pnl",
    "status",
    "resolved_at",
]
MATCH_FRAME_COLUMNS = [
    "match_key",
    "tour",
    "match_date",
    "tournament_key",
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tournament_level",
    "source",
    "source_file",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "loser_id",
    "loser_name",
    "score",
    "best_of",
    "round",
    "winner_sets_won",
    "loser_sets_won",
    "total_games",
    "is_retirement",
    "is_walkover",
    "is_training_eligible",
    "year",
    "days_since_epoch",
]
MATCH_COLUMN_ALIASES = {
    "tourney_level": "tournament_level",
}


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text if text else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except Exception:
        return None


def _safe_bool_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except Exception:
        pass
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y"} else 0
    return 1 if bool(value) else 0


def _row_json(row: dict[str, Any]) -> str:
    clean: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (pd.Timestamp, datetime)):
            clean[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")
            continue
        try:
            if pd.isna(value):
                clean[key] = None
                continue
        except Exception:
            pass
        clean[key] = value.item() if hasattr(value, "item") else value
    return json.dumps(clean, ensure_ascii=True, default=str, separators=(",", ":"))


def _normalize_tournament_key(tour: str | None, tourney_id: str | None, tournament_name: str | None) -> str | None:
    parts = [part for part in [tour, tourney_id, normalize_player_name(tournament_name)] if part]
    if not parts:
        return None
    return "|".join(parts)


def _snapshot_key(row: dict[str, Any]) -> str:
    parts = [
        _safe_text(row.get("match_id")),
        _safe_text(row.get("captured_at")),
        _safe_text(row.get("match_date")),
        _safe_text(row.get("player_1")),
        _safe_text(row.get("player_2")),
    ]
    return "|".join(part or "" for part in parts)


def _event_id(row: dict[str, Any]) -> str:
    parts = [
        _safe_text(row.get("timestamp")),
        _safe_text(row.get("prediction_id")),
        _safe_text(row.get("note")),
        _safe_text(row.get("capital_after")),
    ]
    return "|".join(part or "" for part in parts)


def _elo_rating_key(tour: str, row: dict[str, Any]) -> str:
    parts = [
        tour,
        _safe_text(row.get("match_key")),
        _safe_text(row.get("player_id")),
        _safe_text(row.get("result")),
    ]
    return "|".join(part or "" for part in parts)


def _elo_snapshot_key(tour: str, row: dict[str, Any]) -> str:
    parts = [
        tour,
        _safe_text(row.get("player_id")),
        _safe_text(row.get("surface")),
    ]
    return "|".join(part or "" for part in parts)


def _feature_key(tour: str, row: dict[str, Any]) -> str:
    parts = [
        tour,
        _safe_text(row.get("match_key")),
    ]
    return "|".join(part or "" for part in parts)


def get_connection(db_path: Path = SQLITE_DB_FILE) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def managed_connection(db_path: Path = SQLITE_DB_FILE) -> Any:
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def initialize_database(db_path: Path = SQLITE_DB_FILE) -> Path:
    with managed_connection(db_path) as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        for table_name, columns in SCHEMA_COLUMN_MIGRATIONS.items():
            existing_columns = {
                str(row["name"])
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            for column_name, column_type in columns.items():
                if column_name not in existing_columns:
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        conn.commit()
    return db_path


def _safe_read_fallback_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=columns or [])
    if columns is None:
        return df
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def load_odds_frame(
    *,
    current_only: bool,
    db_path: Path = SQLITE_DB_FILE,
    fallback_to_csv: bool = True,
) -> pd.DataFrame:
    initialize_database(db_path)
    query = """
        SELECT
            match_date,
            match_time,
            tour,
            tournament,
            surface,
            player_1,
            player_2,
            player_1_resolved,
            player_2_resolved,
            player_1_id,
            player_2_id,
            player_1_match_score,
            player_2_match_score,
            odds_p1,
            odds_p2,
            bookmaker,
            bookmaker_count,
            aggregation_method,
            source_url,
            match_id,
            captured_at
        FROM odds_snapshots
    """
    params: tuple[Any, ...] = ()
    if current_only:
        query += " WHERE is_current_upcoming = 1"
    query += " ORDER BY match_date, match_time, captured_at"

    with managed_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if not df.empty:
        return df
    if not fallback_to_csv:
        return pd.DataFrame(columns=ODDS_FRAME_COLUMNS)
    fallback_path = ODDS_UPCOMING_FILE if current_only else ODDS_HISTORY_FILE
    return _safe_read_fallback_csv(fallback_path, ODDS_FRAME_COLUMNS)


def load_prediction_log_frame(
    db_path: Path = SQLITE_DB_FILE,
    *,
    fallback_to_csv: bool = True,
) -> pd.DataFrame:
    initialize_database(db_path)
    query = """
        SELECT
            prediction_id,
            created_at,
            match_date,
            tour,
            match,
            p1_name,
            p2_name,
            bet_side,
            probability,
            odds,
            edge,
            stake,
            confidence_tier,
            model_agreement,
            result,
            pnl,
            status,
            resolved_at
        FROM predictions
        ORDER BY COALESCE(created_at, ''), prediction_id
    """
    with managed_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    if not df.empty:
        return df
    if not fallback_to_csv:
        return pd.DataFrame(columns=PREDICTION_FRAME_COLUMNS)
    return _safe_read_fallback_csv(META_DIR / "prediction_log.csv", PREDICTION_FRAME_COLUMNS)


def load_bankroll_state_payload(
    db_path: Path = SQLITE_DB_FILE,
    *,
    fallback_to_file: bool = True,
) -> dict[str, Any]:
    initialize_database(db_path)
    with managed_connection(db_path) as conn:
        row = conn.execute(
            "SELECT raw_payload_json FROM bankroll_state WHERE state_key = ?",
            ("current",),
        ).fetchone()
    if row and row["raw_payload_json"]:
        try:
            payload = json.loads(row["raw_payload_json"])
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    if fallback_to_file and BANKROLL_LOG_FILE.exists():
        try:
            raw = BANKROLL_LOG_FILE.read_text(encoding="utf-8-sig")
            payload = json.loads(raw) if raw.strip() else {}
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def load_matches_frame(
    tour: str,
    *,
    columns: list[str] | None = None,
    training_eligible_only: bool = False,
    db_path: Path = SQLITE_DB_FILE,
    fallback_to_csv: bool = True,
) -> pd.DataFrame:
    initialize_database(db_path)

    requested_columns = list(columns or MATCH_FRAME_COLUMNS)
    missing_from_db = [col for col in requested_columns if col not in MATCH_FRAME_COLUMNS and col not in MATCH_COLUMN_ALIASES]

    select_parts: list[str] = []
    seen_targets: set[str] = set()
    for col in requested_columns:
        if col in MATCH_COLUMN_ALIASES:
            source_col = MATCH_COLUMN_ALIASES[col]
            select_parts.append(f"{source_col} AS {col}")
            seen_targets.add(source_col)
        elif col in MATCH_FRAME_COLUMNS:
            select_parts.append(col)
            seen_targets.add(col)

    if missing_from_db and "raw_payload_json" not in seen_targets:
        select_parts.append("raw_payload_json")

    if not select_parts:
        select_parts = ["match_key", "raw_payload_json"]

    query = f"""
        SELECT {", ".join(select_parts)}
        FROM matches
        WHERE lower(tour) = ?
    """
    params: list[Any] = [tour.lower()]
    if training_eligible_only:
        query += " AND is_training_eligible = 1"
    query += " ORDER BY match_date, tourney_id, match_num"

    with managed_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=tuple(params))

    if df.empty:
        if not fallback_to_csv:
            return pd.DataFrame(columns=requested_columns)
        path = PROCESSED_DIR / f"{tour}_matches_master.csv"
        return _safe_read_fallback_csv(path, requested_columns)

    if missing_from_db:
        raw_rows = df.get("raw_payload_json", pd.Series(dtype="object")).fillna("{}")
        payloads = raw_rows.map(
            lambda value: json.loads(value) if isinstance(value, str) and value.strip() else {}
        )
        for col in missing_from_db:
            df[col] = payloads.map(lambda payload: payload.get(col) if isinstance(payload, dict) else None)

    for col in requested_columns:
        if col not in df.columns:
            df[col] = pd.NA

    return df[requested_columns]


def latest_match_date(
    *,
    tour: str | None = None,
    db_path: Path = SQLITE_DB_FILE,
    fallback_to_csv: bool = True,
) -> str | None:
    initialize_database(db_path)
    query = "SELECT MAX(match_date) AS latest_match_date FROM matches"
    params: tuple[Any, ...] = ()
    if tour:
        query += " WHERE lower(tour) = ?"
        params = (tour.lower(),)
    with managed_connection(db_path) as conn:
        row = conn.execute(query, params).fetchone()

    latest = _safe_text(row["latest_match_date"]) if row is not None else None
    if latest:
        return latest

    if not fallback_to_csv:
        return None

    if tour:
        path = PROCESSED_DIR / f"{tour}_matches_master.csv"
        fallback_df = _safe_read_fallback_csv(path, ["match_date"])
        if fallback_df.empty:
            return None
        parsed = pd.to_datetime(fallback_df["match_date"], errors="coerce").dropna()
        return parsed.max().strftime("%Y-%m-%d") if not parsed.empty else None

    dates = [
        latest_match_date(tour=single_tour, db_path=db_path, fallback_to_csv=fallback_to_csv)
        for single_tour in ("atp", "wta")
    ]
    parsed_dates = pd.to_datetime([value for value in dates if value], errors="coerce").dropna()
    return parsed_dates.max().strftime("%Y-%m-%d") if not parsed_dates.empty else None


def count_matches(
    *,
    tour: str | None = None,
    db_path: Path = SQLITE_DB_FILE,
    fallback_to_csv: bool = True,
) -> int:
    initialize_database(db_path)
    query = "SELECT COUNT(*) AS row_count FROM matches"
    params: tuple[Any, ...] = ()
    if tour:
        query += " WHERE lower(tour) = ?"
        params = (tour.lower(),)
    with managed_connection(db_path) as conn:
        row = conn.execute(query, params).fetchone()
    if row is not None and int(row["row_count"]) > 0:
        return int(row["row_count"])
    if not fallback_to_csv or not tour:
        return int(row["row_count"]) if row is not None else 0
    path = PROCESSED_DIR / f"{tour}_matches_master.csv"
    return int(len(_safe_read_fallback_csv(path)))


def _record_ingest_run(
    conn: sqlite3.Connection,
    source_name: str,
    rows_synced: int,
    detail: dict[str, Any] | None = None,
) -> None:
    now = _utc_now()
    conn.execute(
        """
        INSERT INTO ingest_runs(source_name, last_synced_at, rows_synced, detail_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(source_name) DO UPDATE SET
            last_synced_at = excluded.last_synced_at,
            rows_synced = excluded.rows_synced,
            detail_json = excluded.detail_json
        """,
        (source_name, now, int(rows_synced), json.dumps(detail or {}, ensure_ascii=True)),
    )


def sync_reference_players(db_path: Path = SQLITE_DB_FILE) -> dict[str, Any]:
    initialize_database(db_path)
    synced = 0
    seen_ids: set[str] = set()
    now = _utc_now()

    with managed_connection(db_path) as conn:
        for tour, path in (("atp", RAW_ATP / "atp_players.csv"), ("wta", RAW_WTA / "wta_players.csv")):
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=["player_id", "name_first", "name_last", "ioc"], low_memory=False)
            except Exception:
                continue
            for row in df.to_dict(orient="records"):
                player_id = _safe_text(row.get("player_id"))
                first = _safe_text(row.get("name_first"))
                last = _safe_text(row.get("name_last"))
                if not player_id or player_id in seen_ids:
                    continue
                canonical_name = " ".join(part for part in [first, last] if part).strip()
                if not canonical_name:
                    continue
                seen_ids.add(player_id)
                conn.execute(
                    """
                    INSERT INTO players(player_id, canonical_name, normalized_name, tour, ioc, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_id) DO UPDATE SET
                        canonical_name = excluded.canonical_name,
                        normalized_name = excluded.normalized_name,
                        tour = excluded.tour,
                        ioc = COALESCE(excluded.ioc, players.ioc),
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    (
                        player_id,
                        canonical_name,
                        normalize_player_name(canonical_name),
                        tour,
                        _safe_text(row.get("ioc")),
                        "reference_csv",
                        now,
                    ),
                )
                synced += 1
        _record_ingest_run(conn, "reference_players", synced)
        conn.commit()

    return {"source": "reference_players", "rows_synced": synced, "db_path": str(db_path)}


def sync_player_aliases(db_path: Path = SQLITE_DB_FILE, aliases: dict[str, str] | None = None) -> dict[str, Any]:
    initialize_database(db_path)
    alias_map = aliases if aliases is not None else load_player_aliases()
    now = _utc_now()

    with managed_connection(db_path) as conn:
        rows = 0
        for alias, canonical in alias_map.items():
            conn.execute(
                """
                INSERT INTO player_aliases(alias, normalized_alias, canonical_name, normalized_canonical_name, player_id, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(alias) DO UPDATE SET
                    normalized_alias = excluded.normalized_alias,
                    canonical_name = excluded.canonical_name,
                    normalized_canonical_name = excluded.normalized_canonical_name,
                    player_id = COALESCE(excluded.player_id, player_aliases.player_id),
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    alias,
                    normalize_player_name(alias),
                    canonical,
                    normalize_player_name(canonical),
                    None,
                    "player_aliases_json",
                    now,
                ),
            )
            rows += 1
        _record_ingest_run(conn, "player_aliases", rows)
        conn.commit()

    return {"source": "player_aliases", "rows_synced": rows, "db_path": str(db_path)}


def sync_matches_frame(df: pd.DataFrame, tour: str, db_path: Path = SQLITE_DB_FILE) -> dict[str, Any]:
    initialize_database(db_path)
    if df.empty:
        return {"source": f"matches_{tour}", "rows_synced": 0, "db_path": str(db_path)}

    now = _utc_now()
    with managed_connection(db_path) as conn:
        rows = 0
        for row in df.to_dict(orient="records"):
            tournament_name = _safe_text(row.get("tourney_name"))
            tournament_key = _normalize_tournament_key(tour, _safe_text(row.get("tourney_id")), tournament_name)
            if tournament_key and tournament_name:
                conn.execute(
                    """
                    INSERT INTO tournaments(tournament_key, tour, tournament_name, normalized_name, surface, tournament_level, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tournament_key) DO UPDATE SET
                        tournament_name = excluded.tournament_name,
                        normalized_name = excluded.normalized_name,
                        surface = COALESCE(excluded.surface, tournaments.surface),
                        tournament_level = COALESCE(excluded.tournament_level, tournaments.tournament_level),
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    (
                        tournament_key,
                        tour,
                        tournament_name,
                        normalize_player_name(tournament_name),
                        _safe_text(row.get("surface")),
                        _safe_text(row.get("tournament_level") or row.get("tourney_level")),
                        _safe_text(row.get("source")) or "matches_master",
                        now,
                    ),
                )

            for player_id_key, player_name_key in (("winner_id", "winner_name"), ("loser_id", "loser_name")):
                player_id = _safe_text(row.get(player_id_key))
                player_name = _safe_text(row.get(player_name_key))
                if not player_id or not player_name:
                    continue
                conn.execute(
                    """
                    INSERT INTO players(player_id, canonical_name, normalized_name, tour, ioc, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_id) DO UPDATE SET
                        canonical_name = excluded.canonical_name,
                        normalized_name = excluded.normalized_name,
                        tour = COALESCE(excluded.tour, players.tour),
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    (
                        player_id,
                        player_name,
                        normalize_player_name(player_name),
                        tour,
                        None,
                        _safe_text(row.get("source")) or "matches_master",
                        now,
                    ),
                )

            match_key = _safe_text(row.get("match_key"))
            if not match_key:
                continue
            conn.execute(
                """
                INSERT INTO matches(
                    match_key, tour, match_date, tournament_key, tourney_id, tourney_name, surface, draw_size,
                    tournament_level, source, source_file, tourney_date, match_num, winner_id, winner_name,
                    loser_id, loser_name, score, best_of, round, winner_sets_won, loser_sets_won, total_games,
                    is_retirement, is_walkover, is_training_eligible, year, days_since_epoch, raw_payload_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_key) DO UPDATE SET
                    tour = excluded.tour,
                    match_date = excluded.match_date,
                    tournament_key = excluded.tournament_key,
                    tourney_id = excluded.tourney_id,
                    tourney_name = excluded.tourney_name,
                    surface = excluded.surface,
                    draw_size = excluded.draw_size,
                    tournament_level = excluded.tournament_level,
                    source = excluded.source,
                    source_file = excluded.source_file,
                    tourney_date = excluded.tourney_date,
                    match_num = excluded.match_num,
                    winner_id = excluded.winner_id,
                    winner_name = excluded.winner_name,
                    loser_id = excluded.loser_id,
                    loser_name = excluded.loser_name,
                    score = excluded.score,
                    best_of = excluded.best_of,
                    round = excluded.round,
                    winner_sets_won = excluded.winner_sets_won,
                    loser_sets_won = excluded.loser_sets_won,
                    total_games = excluded.total_games,
                    is_retirement = excluded.is_retirement,
                    is_walkover = excluded.is_walkover,
                    is_training_eligible = excluded.is_training_eligible,
                    year = excluded.year,
                    days_since_epoch = excluded.days_since_epoch,
                    raw_payload_json = excluded.raw_payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    match_key,
                    tour,
                    _safe_text(row.get("match_date")),
                    tournament_key,
                    _safe_text(row.get("tourney_id")),
                    tournament_name,
                    _safe_text(row.get("surface")),
                    _safe_int(row.get("draw_size")),
                    _safe_text(row.get("tournament_level") or row.get("tourney_level")),
                    _safe_text(row.get("source")) or "matches_master",
                    _safe_text(row.get("source_file")),
                    _safe_text(row.get("tourney_date")),
                    _safe_text(row.get("match_num")),
                    _safe_text(row.get("winner_id")),
                    _safe_text(row.get("winner_name")),
                    _safe_text(row.get("loser_id")),
                    _safe_text(row.get("loser_name")),
                    _safe_text(row.get("score")),
                    _safe_text(row.get("best_of")),
                    _safe_text(row.get("round")),
                    _safe_int(row.get("winner_sets_won")),
                    _safe_int(row.get("loser_sets_won")),
                    _safe_int(row.get("total_games")),
                    _safe_bool_int(row.get("is_retirement")),
                    _safe_bool_int(row.get("is_walkover")),
                    _safe_bool_int(row.get("is_training_eligible")),
                    _safe_int(row.get("year")),
                    _safe_int(row.get("days_since_epoch")),
                    _row_json(row),
                    now,
                ),
            )
            rows += 1

        _record_ingest_run(conn, f"matches_{tour}", rows)
        conn.commit()

    return {"source": f"matches_{tour}", "rows_synced": rows, "db_path": str(db_path)}


def sync_elo_ratings(
    ratings_df: pd.DataFrame,
    *,
    tour: str,
    snapshot_df: pd.DataFrame | None = None,
    db_path: Path = SQLITE_DB_FILE,
) -> dict[str, Any]:
    initialize_database(db_path)
    if ratings_df.empty and (snapshot_df is None or snapshot_df.empty):
        return {
            "source": f"elo_{tour}",
            "rows_synced": 0,
            "snapshot_rows_synced": 0,
            "db_path": str(db_path),
        }

    now = _utc_now()
    with managed_connection(db_path) as conn:
        rating_rows = 0
        for row in ratings_df.to_dict(orient="records"):
            rating_key = _elo_rating_key(tour, row)
            if not rating_key.strip("|"):
                continue
            conn.execute(
                """
                INSERT INTO elo_ratings(
                    rating_key, tour, match_key, match_date, player_id, player_name, opponent_id, opponent_name,
                    result, surface, tourney_id, tourney_name, tourney_level, round, k_factor, elo_pre, elo_post,
                    surface_elo_pre, surface_elo_post, raw_payload_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rating_key) DO UPDATE SET
                    tour = excluded.tour,
                    match_key = excluded.match_key,
                    match_date = excluded.match_date,
                    player_id = excluded.player_id,
                    player_name = excluded.player_name,
                    opponent_id = excluded.opponent_id,
                    opponent_name = excluded.opponent_name,
                    result = excluded.result,
                    surface = excluded.surface,
                    tourney_id = excluded.tourney_id,
                    tourney_name = excluded.tourney_name,
                    tourney_level = excluded.tourney_level,
                    round = excluded.round,
                    k_factor = excluded.k_factor,
                    elo_pre = excluded.elo_pre,
                    elo_post = excluded.elo_post,
                    surface_elo_pre = excluded.surface_elo_pre,
                    surface_elo_post = excluded.surface_elo_post,
                    raw_payload_json = excluded.raw_payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    rating_key,
                    tour,
                    _safe_text(row.get("match_key")),
                    _safe_text(row.get("match_date")),
                    _safe_text(row.get("player_id")),
                    _safe_text(row.get("player_name")),
                    _safe_text(row.get("opponent_id")),
                    _safe_text(row.get("opponent_name")),
                    _safe_int(row.get("result")),
                    _safe_text(row.get("surface")),
                    _safe_text(row.get("tourney_id")),
                    _safe_text(row.get("tourney_name")),
                    _safe_text(row.get("tourney_level")),
                    _safe_text(row.get("round")),
                    _safe_float(row.get("k_factor")),
                    _safe_float(row.get("elo_pre")),
                    _safe_float(row.get("elo_post")),
                    _safe_float(row.get("surface_elo_pre")),
                    _safe_float(row.get("surface_elo_post")),
                    _row_json(row),
                    now,
                ),
            )
            rating_rows += 1

        snapshot_rows = 0
        if snapshot_df is not None and not snapshot_df.empty:
            for row in snapshot_df.to_dict(orient="records"):
                snapshot_key = _elo_snapshot_key(tour, row)
                if not snapshot_key.strip("|"):
                    continue
                conn.execute(
                    """
                    INSERT INTO elo_snapshot(snapshot_key, tour, player_id, surface, elo, raw_payload_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_key) DO UPDATE SET
                        tour = excluded.tour,
                        player_id = excluded.player_id,
                        surface = excluded.surface,
                        elo = excluded.elo,
                        raw_payload_json = excluded.raw_payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        snapshot_key,
                        tour,
                        _safe_text(row.get("player_id")),
                        _safe_text(row.get("surface")),
                        _safe_float(row.get("elo")) or 0.0,
                        _row_json(row),
                        now,
                    ),
                )
                snapshot_rows += 1

        _record_ingest_run(
            conn,
            f"elo_{tour}",
            rating_rows,
            {"snapshot_rows_synced": snapshot_rows},
        )
        conn.commit()

    return {
        "source": f"elo_{tour}",
        "rows_synced": rating_rows,
        "snapshot_rows_synced": snapshot_rows,
        "db_path": str(db_path),
    }


def sync_features_frame(df: pd.DataFrame, *, tour: str, db_path: Path = SQLITE_DB_FILE) -> dict[str, Any]:
    initialize_database(db_path)
    if df.empty:
        return {"source": f"features_{tour}", "rows_synced": 0, "db_path": str(db_path)}

    now = _utc_now()
    with managed_connection(db_path) as conn:
        rows = 0
        for row in df.to_dict(orient="records"):
            feature_key = _feature_key(tour, row)
            if not feature_key.strip("|"):
                continue
            conn.execute(
                """
                INSERT INTO player_features(
                    feature_key, tour, match_key, match_date, p1_id, p2_id, p1_name, p2_name, surface,
                    tournament_level, round, best_of, p1_wins, raw_payload_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(feature_key) DO UPDATE SET
                    tour = excluded.tour,
                    match_key = excluded.match_key,
                    match_date = excluded.match_date,
                    p1_id = excluded.p1_id,
                    p2_id = excluded.p2_id,
                    p1_name = excluded.p1_name,
                    p2_name = excluded.p2_name,
                    surface = excluded.surface,
                    tournament_level = excluded.tournament_level,
                    round = excluded.round,
                    best_of = excluded.best_of,
                    p1_wins = excluded.p1_wins,
                    raw_payload_json = excluded.raw_payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    feature_key,
                    tour,
                    _safe_text(row.get("match_key")),
                    _safe_text(row.get("match_date")),
                    _safe_text(row.get("p1_id")),
                    _safe_text(row.get("p2_id")),
                    _safe_text(row.get("p1_name")),
                    _safe_text(row.get("p2_name")),
                    _safe_text(row.get("surface")),
                    _safe_text(row.get("tournament_level")),
                    _safe_text(row.get("round")),
                    _safe_text(row.get("best_of")),
                    _safe_int(row.get("p1_wins")),
                    _row_json(row),
                    now,
                ),
            )
            rows += 1

        _record_ingest_run(conn, f"features_{tour}", rows)
        conn.commit()

    return {"source": f"features_{tour}", "rows_synced": rows, "db_path": str(db_path)}


def sync_odds_frame(
    df: pd.DataFrame,
    *,
    mark_current_upcoming: bool,
    source_name: str,
    db_path: Path = SQLITE_DB_FILE,
) -> dict[str, Any]:
    initialize_database(db_path)
    if df.empty:
        return {"source": source_name, "rows_synced": 0, "db_path": str(db_path)}

    now = _utc_now()
    with managed_connection(db_path) as conn:
        if mark_current_upcoming:
            conn.execute("UPDATE odds_snapshots SET is_current_upcoming = 0 WHERE is_current_upcoming = 1")

        rows = 0
        for row in df.to_dict(orient="records"):
            for player_id_key, player_name_key in (("player_1_id", "player_1_resolved"), ("player_2_id", "player_2_resolved")):
                player_id = _safe_text(row.get(player_id_key))
                player_name = _safe_text(row.get(player_name_key))
                if not player_id or not player_name:
                    continue
                conn.execute(
                    """
                    INSERT INTO players(player_id, canonical_name, normalized_name, tour, ioc, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_id) DO UPDATE SET
                        canonical_name = excluded.canonical_name,
                        normalized_name = excluded.normalized_name,
                        tour = COALESCE(excluded.tour, players.tour),
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    (
                        player_id,
                        player_name,
                        normalize_player_name(player_name),
                        _safe_text(row.get("tour")),
                        None,
                        source_name,
                        now,
                    ),
                )

            snapshot_key = _snapshot_key(row)
            conn.execute(
                """
                INSERT INTO odds_snapshots(
                    snapshot_key, match_id, match_date, match_time, tour, tournament, surface, player_1, player_2,
                    player_1_resolved, player_2_resolved, player_1_id, player_2_id, player_1_match_score,
                    player_2_match_score, odds_p1, odds_p2, bookmaker, bookmaker_count, aggregation_method,
                    source_url, captured_at, is_current_upcoming, raw_payload_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_key) DO UPDATE SET
                    match_id = excluded.match_id,
                    match_date = excluded.match_date,
                    match_time = excluded.match_time,
                    tour = excluded.tour,
                    tournament = excluded.tournament,
                    surface = excluded.surface,
                    player_1 = excluded.player_1,
                    player_2 = excluded.player_2,
                    player_1_resolved = excluded.player_1_resolved,
                    player_2_resolved = excluded.player_2_resolved,
                    player_1_id = excluded.player_1_id,
                    player_2_id = excluded.player_2_id,
                    player_1_match_score = excluded.player_1_match_score,
                    player_2_match_score = excluded.player_2_match_score,
                    odds_p1 = excluded.odds_p1,
                    odds_p2 = excluded.odds_p2,
                    bookmaker = excluded.bookmaker,
                    bookmaker_count = excluded.bookmaker_count,
                    aggregation_method = excluded.aggregation_method,
                    source_url = excluded.source_url,
                    captured_at = excluded.captured_at,
                    is_current_upcoming = excluded.is_current_upcoming,
                    raw_payload_json = excluded.raw_payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    snapshot_key,
                    _safe_text(row.get("match_id")),
                    _safe_text(row.get("match_date")),
                    _safe_text(row.get("match_time")),
                    _safe_text(row.get("tour")),
                    _safe_text(row.get("tournament")),
                    _safe_text(row.get("surface")),
                    _safe_text(row.get("player_1")),
                    _safe_text(row.get("player_2")),
                    _safe_text(row.get("player_1_resolved")),
                    _safe_text(row.get("player_2_resolved")),
                    _safe_text(row.get("player_1_id")),
                    _safe_text(row.get("player_2_id")),
                    _safe_float(row.get("player_1_match_score")),
                    _safe_float(row.get("player_2_match_score")),
                    _safe_float(row.get("odds_p1")),
                    _safe_float(row.get("odds_p2")),
                    _safe_text(row.get("bookmaker")),
                    _safe_int(row.get("bookmaker_count")),
                    _safe_text(row.get("aggregation_method")),
                    _safe_text(row.get("source_url")),
                    _safe_text(row.get("captured_at")),
                    1 if mark_current_upcoming else 0,
                    _row_json(row),
                    now,
                ),
            )
            rows += 1

        _record_ingest_run(conn, source_name, rows, {"mark_current_upcoming": mark_current_upcoming})
        conn.commit()

    return {"source": source_name, "rows_synced": rows, "db_path": str(db_path)}


def sync_prediction_log_frame(df: pd.DataFrame, db_path: Path = SQLITE_DB_FILE) -> dict[str, Any]:
    initialize_database(db_path)
    if df.empty:
        return {"source": "prediction_log", "rows_synced": 0, "db_path": str(db_path)}

    now = _utc_now()
    with managed_connection(db_path) as conn:
        rows = 0
        for row in df.to_dict(orient="records"):
            prediction_id = _safe_text(row.get("prediction_id"))
            if not prediction_id:
                continue
            conn.execute(
                """
                INSERT INTO predictions(
                    prediction_id, created_at, match_date, tour, match, p1_name, p2_name, bet_side, probability,
                    odds, edge, stake, confidence_tier, model_agreement, result, pnl, status, resolved_at,
                    raw_payload_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prediction_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    match_date = excluded.match_date,
                    tour = excluded.tour,
                    match = excluded.match,
                    p1_name = excluded.p1_name,
                    p2_name = excluded.p2_name,
                    bet_side = excluded.bet_side,
                    probability = excluded.probability,
                    odds = excluded.odds,
                    edge = excluded.edge,
                    stake = excluded.stake,
                    confidence_tier = excluded.confidence_tier,
                    model_agreement = excluded.model_agreement,
                    result = excluded.result,
                    pnl = excluded.pnl,
                    status = excluded.status,
                    resolved_at = excluded.resolved_at,
                    raw_payload_json = excluded.raw_payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    prediction_id,
                    _safe_text(row.get("created_at")),
                    _safe_text(row.get("match_date")),
                    _safe_text(row.get("tour")),
                    _safe_text(row.get("match")),
                    _safe_text(row.get("p1_name")),
                    _safe_text(row.get("p2_name")),
                    _safe_text(row.get("bet_side")),
                    _safe_float(row.get("probability")),
                    _safe_float(row.get("odds")),
                    _safe_float(row.get("edge")),
                    _safe_float(row.get("stake")),
                    _safe_text(row.get("confidence_tier")),
                    _safe_float(row.get("model_agreement")),
                    _safe_text(row.get("result")),
                    _safe_float(row.get("pnl")),
                    _safe_text(row.get("status")),
                    _safe_text(row.get("resolved_at")),
                    _row_json(row),
                    now,
                ),
            )
            rows += 1

        _record_ingest_run(conn, "prediction_log", rows)
        conn.commit()

    return {"source": "prediction_log", "rows_synced": rows, "db_path": str(db_path)}


def sync_bankroll_state(state: dict[str, Any], db_path: Path = SQLITE_DB_FILE) -> dict[str, Any]:
    initialize_database(db_path)
    now = _utc_now()
    history = state.get("history", [])
    if not isinstance(history, list):
        history = []

    with managed_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO bankroll_state(state_key, capital, updated_at, history_count, raw_payload_json, synced_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                capital = excluded.capital,
                updated_at = excluded.updated_at,
                history_count = excluded.history_count,
                raw_payload_json = excluded.raw_payload_json,
                synced_at = excluded.synced_at
            """,
            (
                "current",
                float(state.get("capital", 0.0) or 0.0),
                _safe_text(state.get("updated_at")),
                len(history),
                json.dumps(state, ensure_ascii=True),
                now,
            ),
        )

        rows = 0
        for row in history:
            if not isinstance(row, dict):
                continue
            conn.execute(
                """
                INSERT INTO bankroll_events(
                    event_id, timestamp, prediction_id, note, pnl, capital_before, capital_after, raw_payload_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    prediction_id = excluded.prediction_id,
                    note = excluded.note,
                    pnl = excluded.pnl,
                    capital_before = excluded.capital_before,
                    capital_after = excluded.capital_after,
                    raw_payload_json = excluded.raw_payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    _event_id(row),
                    _safe_text(row.get("timestamp")),
                    _safe_text(row.get("prediction_id")),
                    _safe_text(row.get("note")),
                    _safe_float(row.get("pnl")),
                    _safe_float(row.get("capital_before")),
                    _safe_float(row.get("capital_after")),
                    _row_json(row),
                    now,
                ),
            )
            rows += 1

        _record_ingest_run(conn, "bankroll", rows, {"capital": float(state.get("capital", 0.0) or 0.0)})
        conn.commit()

    return {"source": "bankroll", "rows_synced": rows, "db_path": str(db_path)}


def bootstrap_sqlite_from_files(db_path: Path = SQLITE_DB_FILE) -> dict[str, Any]:
    initialize_database(db_path)
    results: dict[str, Any] = {
        "db_path": str(db_path),
        "sources": {},
    }

    results["sources"]["reference_players"] = sync_reference_players(db_path=db_path)
    results["sources"]["player_aliases"] = sync_player_aliases(db_path=db_path)

    for tour in ("atp", "wta"):
        path = PROCESSED_DIR / f"{tour}_matches_master.csv"
        if path.exists() and path.stat().st_size > 0:
            try:
                df = pd.read_csv(path, low_memory=False)
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
        results["sources"][f"matches_{tour}"] = sync_matches_frame(df, tour=tour, db_path=db_path)

        elo_path = PROCESSED_DIR / f"{tour}_elo_ratings.csv"
        if elo_path.exists() and elo_path.stat().st_size > 0:
            try:
                elo_df = pd.read_csv(elo_path, low_memory=False)
            except Exception:
                elo_df = pd.DataFrame()
        else:
            elo_df = pd.DataFrame()

        snapshot_path = PROCESSED_DIR / f"{tour}_elo_snapshot.csv"
        if snapshot_path.exists() and snapshot_path.stat().st_size > 0:
            try:
                snapshot_df = pd.read_csv(snapshot_path, low_memory=False)
            except Exception:
                snapshot_df = pd.DataFrame()
        else:
            snapshot_df = pd.DataFrame()
        results["sources"][f"elo_{tour}"] = sync_elo_ratings(
            elo_df,
            tour=tour,
            snapshot_df=snapshot_df,
            db_path=db_path,
        )

        feature_path = PROCESSED_DIR / f"{tour}_player_features.csv"
        if feature_path.exists() and feature_path.stat().st_size > 0:
            try:
                feature_df = pd.read_csv(feature_path, low_memory=False)
            except Exception:
                feature_df = pd.DataFrame()
        else:
            feature_df = pd.DataFrame()
        results["sources"][f"features_{tour}"] = sync_features_frame(feature_df, tour=tour, db_path=db_path)

    if ODDS_HISTORY_FILE.exists() and ODDS_HISTORY_FILE.stat().st_size > 0:
        try:
            history_df = pd.read_csv(ODDS_HISTORY_FILE, low_memory=False)
        except Exception:
            history_df = pd.DataFrame()
    else:
        history_df = pd.DataFrame()
    results["sources"]["odds_history"] = sync_odds_frame(
        history_df,
        mark_current_upcoming=False,
        source_name="odds_history",
        db_path=db_path,
    )

    if ODDS_UPCOMING_FILE.exists() and ODDS_UPCOMING_FILE.stat().st_size > 0:
        try:
            upcoming_df = pd.read_csv(ODDS_UPCOMING_FILE, low_memory=False)
        except Exception:
            upcoming_df = pd.DataFrame()
    else:
        upcoming_df = pd.DataFrame()
    results["sources"]["odds_upcoming"] = sync_odds_frame(
        upcoming_df,
        mark_current_upcoming=True,
        source_name="odds_upcoming",
        db_path=db_path,
    )

    prediction_log_file = META_DIR / "prediction_log.csv"
    if prediction_log_file.exists() and prediction_log_file.stat().st_size > 0:
        try:
            prediction_df = pd.read_csv(prediction_log_file, low_memory=False)
        except Exception:
            prediction_df = pd.DataFrame()
    else:
        prediction_df = pd.DataFrame()
    results["sources"]["prediction_log"] = sync_prediction_log_frame(prediction_df, db_path=db_path)

    if BANKROLL_LOG_FILE.exists():
        try:
            raw = BANKROLL_LOG_FILE.read_text(encoding="utf-8-sig")
            bankroll_state = json.loads(raw) if raw.strip() else {}
        except Exception:
            bankroll_state = {}
    else:
        bankroll_state = {}
    results["sources"]["bankroll"] = sync_bankroll_state(bankroll_state, db_path=db_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap or refresh the canonical TennisBet SQLite database")
    parser.add_argument("--sync-all", action="store_true", help="Read current CSV/JSON artifacts and upsert them into SQLite")
    args = parser.parse_args()

    if args.sync_all:
        result = bootstrap_sqlite_from_files()
    else:
        db_path = initialize_database()
        result = {"db_path": str(db_path), "initialized": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
