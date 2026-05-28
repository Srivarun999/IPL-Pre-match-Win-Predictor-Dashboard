from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

REQUIRED_MIN_COLUMNS = [
    # flexible mapping will remap
]

def load_matches_flexible(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coerce common date fields
    for c in ["date"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df

class SeasonCatalog:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def seasons_sorted(self) -> List[int]:
        if "season" in self.df.columns:
            vals = sorted([int(x) for x in pd.unique(self.df["season"].dropna())])
        elif "year" in self.df.columns:
            vals = sorted([int(x) for x in pd.unique(self.df["year"].dropna())])
        else:
            vals = []
        return vals

    def teams_in_season(self, season: int) -> List[str]:
        sub = self.df[self.df["season"] == season] if "season" in self.df.columns else self.df[self.df["year"] == season]
        teams = pd.unique(pd.concat([sub["team1"], sub["team2"]], axis=0).dropna())
        return sorted([str(t) for t in teams])

    def venues_in_season(self, season: int) -> List[str]:
        sub = self.df[self.df["season"] == season] if "season" in self.df.columns else self.df[self.df["year"] == season]
        if "venue" in sub.columns:
            return sorted([str(v) for v in pd.unique(sub["venue"].dropna())])
        return []

    def city_for_venue(self, venue: str) -> Optional[str]:
        sub = self.df[self.df["venue"] == venue] if "venue" in self.df.columns else pd.DataFrame()
        if "city" in sub.columns and not sub.empty:
            vals = sub["city"].dropna().value_counts()
            return str(vals.index[0]) if len(vals) > 0 else None
        return None
