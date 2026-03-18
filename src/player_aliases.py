from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import META_DIR

PLAYER_ALIASES_FILE = META_DIR / "player_aliases.json"


def normalize_player_name(name: Any) -> str:
    return " ".join(str(name).lower().replace(".", " ").replace("-", " ").split())


def load_player_aliases(path: Path = PLAYER_ALIASES_FILE) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    raw = json.loads(text) if text.strip() else {}
    aliases: dict[str, str] = {}
    if not isinstance(raw, dict):
        return aliases
    for alias, canonical in raw.items():
        if not isinstance(alias, str) or not isinstance(canonical, str):
            continue
        alias_norm = normalize_player_name(alias)
        canonical_norm = normalize_player_name(canonical)
        if alias_norm and canonical_norm:
            aliases[alias_norm] = canonical_norm
    return aliases


def canonicalize_player_name(name: Any, aliases: dict[str, str] | None = None) -> str:
    normalized = normalize_player_name(name)
    if not normalized:
        return normalized
    if not aliases:
        return normalized
    return aliases.get(normalized, normalized)
