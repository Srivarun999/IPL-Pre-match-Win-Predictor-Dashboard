from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np

from .player_features import PlayerStatsService
from .utils import safe_log1p

MATCH_TYPE_MAP = {"Normal":0, "Qualifier":1, "Eliminator":2, "Final":3}
DAYNIGHT_MAP = {"Day":0, "Night":1}

def build_match_context_features(season: int, venue: str, city: Optional[str], match_type: str, day_or_night: str, team1: str, team2: str) -> Dict[str, Any]:
    return {
        "season": int(season),
        "match_type_enc": MATCH_TYPE_MAP.get(match_type, 0),
        "day_night_enc": DAYNIGHT_MAP.get(day_or_night, 0),
        "venue_hash": hash(venue) % 1000,
        "city_hash": (hash(city) % 1000) if city else 0,
        "team1_hash": hash(team1) % 1000,
        "team2_hash": hash(team2) % 1000,
    }

def build_selected_xi_player_features(ps: PlayerStatsService, season: int, team: str, xi: List[str], opp: str, venue: str) -> Dict[str, float]:
    # Aggregate per-player vectors, return flattened averages/sums for later aggregation
    # Keep list of vectors if needed for detailed aggregation
    vectors = [ps.player_vector(p, season, opp=opp, venue=venue) for p in xi]
    # For team aggregation we return list to aggregator
    return {"_players": vectors, "_count": len(vectors)}

