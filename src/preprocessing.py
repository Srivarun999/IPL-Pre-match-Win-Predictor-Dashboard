from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional, List
import re

TEAM_ALIASES = {
    "rcb": "Royal Challengers Bangalore",
    "royal challengers bangalore": "Royal Challengers Bangalore",
    "royal challengers": "Royal Challengers Bangalore",
    "csk": "Chennai Super Kings",
    "chennai super kings": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "mumbai indians": "Mumbai Indians",
    "kkr": "Kolkata Knight Riders",
    "kolkata knight riders": "Kolkata Knight Riders",
    "dc": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "dd": "Delhi Daredevils",
    "delhi daredevils": "Delhi Daredevils",
    "kxip": "Kings XI Punjab",
    "kings xi punjab": "Kings XI Punjab",
    "rr": "Rajasthan Royals",
    "rajasthan royals": "Rajasthan Royals",
    "srh": "Sunrisers Hyderabad",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "pbks": "Punjab Kings",
    "punjab kings": "Punjab Kings",
}

COMMON_MAPS = [
    {
        "team1": ["team1", "batting_team", "batting_team_pre", "home_team", "Team1"],
        "team2": ["team2", "bowling_team", "bowling_team_pre", "away_team", "Team2"],
        "winner": ["winner", "match_winner", "match_won_by", "win_outcome_team", "match_won_by_team"],
        "venue": ["venue", "ground"],
        "city": ["city"],
        "season": ["season", "Season", "year"],
        "match_type": ["match_type", "stage", "event_stage"],
        "day_or_night": ["day_or_night", "daynight", "match_time", "day_night"],
        "date": ["date", "match_date"],
        "batter": ["batter", "striker", "batsman"],
        "bowler": ["bowler"],
        "runs_batter": ["runs_batter", "batsman_runs", "runs_off_bat"],
        "balls_faced": ["balls_faced"],
        "wicket_kind": ["wicket_kind", "dismissal_kind"],
        "runs_total": ["runs_total", "total_runs"],
        "overs": ["over", "overs"],
        "ball": ["ball", "ball_no"],
        "innings": ["innings", "inning"],
        "bowler_wicket": ["bowler_wicket"],
        "toss_winner": ["toss_winner"],
        "toss_decision": ["toss_decision"],
        "result_type": ["result_type", "result"],
    }
]

class FlexibleColumnMapper:
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    @classmethod
    def infer(cls, df: pd.DataFrame) -> "FlexibleColumnMapper":
        mapping: Dict[str, str] = {}
        lower_cols = {c.lower(): c for c in df.columns}
        for spec in COMMON_MAPS:
            for std, alts in spec.items():
                for a in alts:
                    if a.lower() in lower_cols:
                        mapping[std] = lower_cols[a.lower()]
                        break
        # Fallback explicit if exactly named
        for std in ["team1", "team2", "winner", "venue", "season"]:
            mapping.setdefault(std, std if std in df.columns else None)
        return cls(mapping)

    def remap(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for std, raw in self.mapping.items():
            if raw in df.columns:
                out[std] = df[raw]
        # Normalize season values to integer year if possible
        if "season" not in out.columns and "year" in out.columns:
            out["season"] = out["year"]
        if "season" in out.columns:
            out["season"] = out["season"].astype(str).str.strip()
            # Convert seasons like 2007/08 or 2010/11 to canonical year end (2008, 2011).
            def normalize_season(value: str) -> int:
                if not isinstance(value, str) or not value:
                    return 0
                value = value.strip()
                if "/" in value:
                    parts = [p.strip() for p in value.split("/") if p.strip()]
                    if len(parts) == 2 and parts[0].isdigit():
                        start = int(parts[0])
                        end = parts[1]
                        if end.isdigit():
                            if len(end) == 2:
                                end = int(str(start)[:2] + end)
                            else:
                                end = int(end)
                            return end
                if value.isdigit():
                    return int(value)
                match = pd.Series([value]).str.extract(r"(\d{4})").iloc[0, 0]
                return int(match) if pd.notna(match) else 0
            out["season"] = out["season"].apply(normalize_season)
        for c in ["team1", "team2", "winner", "venue", "city", "match_type", "day_or_night"]:
            if c in out.columns:
                out[c] = out[c].astype(str)

        def canonical_team(name: str) -> str:
            return TEAM_ALIASES.get(name.strip().lower(), name.strip())

        for c in ["team1", "team2", "winner"]:
            if c in out.columns:
                out[c] = out[c].apply(canonical_team)

        if "venue" in out.columns:
            out["venue"] = out["venue"].str.strip()
        if "city" in out.columns:
            out["city"] = out["city"].str.strip()
        return out

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "team1" in d.columns:
        d["team1"] = d["team1"].astype(str).str.strip().apply(lambda x: TEAM_ALIASES.get(x.lower(), x))
    if "team2" in d.columns:
        d["team2"] = d["team2"].astype(str).str.strip().apply(lambda x: TEAM_ALIASES.get(x.lower(), x))
    if "winner" in d.columns:
        d["winner"] = d["winner"].astype(str).str.strip().replace("Unknown", "").apply(lambda x: TEAM_ALIASES.get(x.lower(), x))
    if "day_or_night" in d.columns:
        d["day_or_night"] = d["day_or_night"].str.strip().str.title().replace({"Na": "Day"})
        d.loc[~d["day_or_night"].isin(["Day", "Night"]), "day_or_night"] = "Day"
    if "match_type" in d.columns:
        d["match_type"] = d["match_type"].astype(str).str.title()
        repl = {"Qualifier 1":"Qualifier", "Qualifier 2":"Qualifier"}
        d["match_type"] = d["match_type"].replace(repl)
        d.loc[~d["match_type"].isin(["Normal", "Qualifier", "Eliminator", "Final"]), "match_type"] = "Normal"
    return d
