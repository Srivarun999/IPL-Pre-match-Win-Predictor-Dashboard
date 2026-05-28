from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# We derive cumulative career and season-to-date stats per player up to season (inclusive).
# If your data has match dates, you can refine to pre-match-by-date. Here we use season-level cutoffs.

@dataclass
class PlayerAggregate:
    player: str
    season: int
    team: Optional[str]
    matches: int
    runs: float
    balls: float
    outs: float
    wickets: float
    overs: float
    dots: float
    fours: float
    sixes: float
    role_guess: str

class PlayerStatsService:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Heuristics to assemble per-player metrics from ball-level if present
        self.has_ball_level = all(c in self.df.columns for c in ["batter", "bowler", "runs_batter"])
        self._prepare()

    def _prepare(self):
        d = self.df.copy()
        if "season" not in d.columns:
            if "year" in d.columns:
                d["season"] = d["year"].astype(int)
            else:
                d["season"] = pd.to_datetime(d["date"], errors="coerce").dt.year.fillna(0).astype(int)

        # Appearances per season and team inference
        # Build player event rows from batter and bowler columns
        batter_events = None
        bowler_events = None
        if "batter" in d.columns:
            batter_events = d[["season", "batter", "team1", "team2", "runs_batter", "runs_total"]].copy()
            batter_events.rename(columns={"batter":"player"}, inplace=True)
            batter_events["as_batter"] = 1
        if "bowler" in d.columns:
            bowler_events = d[["season", "bowler", "team1", "team2", "runs_total"]].copy()
            bowler_events.rename(columns={"bowler":"player"}, inplace=True)
            bowler_events["as_bowler"] = 1

        events = []
        if batter_events is not None:
            events.append(batter_events)
        if bowler_events is not None:
            events.append(bowler_events)
        if events:
            ev = pd.concat(events, axis=0, ignore_index=True).dropna(subset=["player"])
        else:
            # Fallback to players not available
            ev = pd.DataFrame(columns=["season", "player", "team1", "team2"])

        # Infer team association per player-season using majority presence among team1/team2
        def infer_team(g):
            # naive: choose the most frequent team among team1/team2 occurrences
            tcounts = pd.concat([g["team1"], g["team2"]]).value_counts()
            return str(tcounts.index[0]) if len(tcounts) > 0 else None

        team_by_player_season = ev.groupby(["player", "season"]).apply(infer_team).reset_index()
        team_by_player_season.columns = ["player", "season", "team_inferred"]
        self.team_by_player_season = team_by_player_season

        # Build per-player batting summary
        if self.has_ball_level:
            b = d.dropna(subset=["batter"]).copy()
            b["is_boundary_4"] = 1 * (b.get("runs_batter", 0) == 4)
            b["is_boundary_6"] = 1 * (b.get("runs_batter", 0) == 6)
            # balls faced proxy: count valid deliveries faced if available else fallback on assumption
            b["bf"] = 1  # treat each row as ball for batter
            batting = b.groupby(["batter", "season"]).agg(
                runs=("runs_batter", "sum"),
                balls=("bf", "sum"),
                fours=("is_boundary_4", "sum"),
                sixes=("is_boundary_6", "sum"),
            ).reset_index().rename(columns={"batter":"player"})
            # Outs proxy: use wicket_kind with player_out if available; else approximate using runs/avg
            if "wicket_kind" in d.columns and "player_out" in d.columns:
                outs_series = d[d["wicket_kind"].notna()].groupby(["season", "player_out"]).size()
                outs = outs_series.reset_index().rename(columns={0:"outs", "player_out":"player"})
            else:
                outs = batting[["player", "season", "runs"]].copy()
                outs["outs"] = np.maximum((outs["runs"] / 25.0).round(), 1.0)  # conservative proxy
                outs = outs[["player", "season", "outs"]]
            batting = batting.merge(outs, on=["player", "season"], how="left")
            batting["outs"] = batting["outs"].fillna((batting["runs"] / 25.0).clip(lower=1.0))

            # Bowling summary
            w = d.dropna(subset=["bowler"]).copy()
            w["valid_ball"] = 1  # approximate valid deliveries
            bowling = w.groupby(["bowler", "season"]).agg(
                balls=("valid_ball", "sum"),
                runs_conceded=("runs_total", "sum"),
            ).reset_index().rename(columns={"bowler":"player"})
            # wickets proxy
            if "wicket_kind" in d.columns and "player_out" in d.columns:
                wkts = d[d["wicket_kind"].notna()].groupby(["season", "bowler"]).size().reset_index().rename(columns={0:"wickets", "bowler":"player"})
            else:
                wkts = bowling.copy()
                wkts["wickets"] = (wkts["balls"] / 24.0)  # naive strike ~24
                wkts = wkts[["player", "season", "wickets"]]
            bowling = bowling.merge(wkts, on=["player", "season"], how="left")
            bowling["overs"] = bowling["balls"] / 6.0
            bowling["dots"] = 0.0  # unavailable robustly; can be estimated if dot ball flag exists

            # Merge batting+bowling
            self.batting = batting
            self.bowling = bowling
        else:
            # Minimal fallback if only match-level exists
            self.batting = pd.DataFrame(columns=["player", "season", "runs", "balls", "fours", "sixes", "outs"])
            self.bowling = pd.DataFrame(columns=["player", "season", "balls", "runs_conceded", "wickets", "overs", "dots"])

        # Role guess: batter if batting runs >> bowling, bowler otherwise, allrounder if both
        def role_from_row(row):
            r = row.get("bat_runs", 0)
            w = row.get("wkts", 0)
            if r >= 400 and w < 10: return "batter"
            if w >= 15 and r < 250: return "bowler"
            if r >= 250 and w >= 10: return "allrounder"
            return "flex"

        # Build per-season combined
        comb = pd.merge(self.batting.rename(columns={"runs":"bat_runs"}), 
                        self.bowling.rename(columns={"wickets":"wkts"}) , 
                        on=["player","season"], how="outer", suffixes=("","_bowl"))
        for c in ["bat_runs","balls","fours","sixes","outs","balls_bowl","runs_conceded","wkts","overs","dots"]:
            if c not in comb.columns:
                comb[c] = 0.0
        temp = comb[["player","season","bat_runs","balls","outs","wkts"]].copy()
        temp["role_guess"] = temp.apply(role_from_row, axis=1)
        self.per_season = comb.merge(temp[["player","season","role_guess"]], on=["player","season"], how="left")

        # Career cumulative to season
        self.per_career = (
            self.per_season
            .sort_values(["player","season"])
            .groupby("player")
            .apply(
                lambda g: g.assign(
                    career_bat_runs = g["bat_runs"].cumsum(),
                    career_bat_balls = g["balls"].cumsum(),
                    career_outs = g["outs"].cumsum(),
                    career_wkts = g["wkts"].cumsum(),
                    career_bowl_balls = g["balls_bowl"].cumsum(),
                    career_overs = g["overs"].cumsum(),
                )
            )
            .reset_index()
        )
        if "level_1" in self.per_career.columns:
            self.per_career = self.per_career.drop(columns=["level_1"])

    def players_for_team_in_season(self, team: str, season: int) -> List[str]:
        # infer from appearance mapping
        m = self.team_by_player_season
        sub = m[(m["season"] == season) & (m["team_inferred"] == team)]
        players = sorted([str(p) for p in sub["player"].unique()])
        return players

    def player_vector(self, player: str, season: int, opp: Optional[str] = None, venue: Optional[str] = None) -> Dict[str, float]:
        # Build generalized player metrics using season-to-date and career-to-date
        ps = self.per_season[self.per_season["player"] == player]
        pc = self.per_career[self.per_career["player"] == player]
        if ps.empty and pc.empty:
            return {
                "bat_runs":0,"bat_balls":0,"bat_sr":0,"bat_avg":0,"boundary_pct":0,
                "wkts":0,"bowl_overs":0,"bowl_eco":0,"bowl_sr":0,"dot_pct":0,
                "matches":0, "role_batter":0, "role_bowler":0, "role_allrounder":0, "role_flex":1,
                "venue_perf":0, "opp_perf":0,
                "batting_form_index":0,"bowling_form_index":0,"allround_index":0,
                "consistency_score":0,"role_strength":0,
                "pp_strength":0,"middle_strength":0,"death_strength":0,
                "venue_adj":0,"opp_adj":0,"season_career_delta":0
            }

        # Season row (current season)
        p_season = ps[ps["season"] == season]
        if p_season.empty:
            # If not found in season, use last available season as proxy (still pre-match)
            p_season = ps.tail(1)
        if p_season.empty and not pc.empty:
            # derive zero
            pass

        def nz(s, c): 
            return float(s[c].values[0]) if c in s.columns and len(s)>0 and pd.notna(s[c].values[0]) else 0.0

        bat_runs = nz(p_season, "bat_runs")
        bat_balls = nz(p_season, "balls")
        outs = nz(p_season, "outs")
        wkts = nz(p_season, "wkts")
        bowl_balls = nz(p_season, "balls_bowl")
        overs = nz(p_season, "overs")
        runs_conc = nz(p_season, "runs_conceded")
        dots = nz(p_season, "dots")

        # Career to date (up to season)
        p_career_upto = pc[pc["season"] <= season]
        if p_career_upto.empty:
            p_career_upto = pc
        c_bat_runs = p_career_upto["career_bat_runs"].max() if "career_bat_runs" in p_career_upto else 0.0
        c_bat_balls = p_career_upto["career_bat_balls"].max() if "career_bat_balls" in p_career_upto else 0.0
        c_outs = p_career_upto["career_outs"].max() if "career_outs" in p_career_upto else 1.0
        c_wkts = p_career_upto["career_wkts"].max() if "career_wkts" in p_career_upto else 0.0
        c_bowl_balls = p_career_upto["career_bowl_balls"].max() if "career_bowl_balls" in p_career_upto else 1.0
        c_overs = p_career_upto["career_overs"].max() if "career_overs" in p_career_upto else 0.0

        # Derived rates
        bat_sr = (bat_runs / bat_balls * 100.0) if bat_balls > 0 else 0.0
        bat_avg = (bat_runs / outs) if outs > 0 else bat_runs
        boundary_pct = 0.0
        bowl_eco = (runs_conc / overs) if overs > 0 else 0.0
        bowl_sr = (bowl_balls / wkts) if wkts > 0 else 0.0
        dot_pct = (dots / bowl_balls) if bowl_balls > 0 else 0.0

        # Role encoding
        role_guess = "flex"
        if not p_season.empty and "role_guess" in p_season.columns and isinstance(p_season["role_guess"].values[0], str):
            role_guess = p_season["role_guess"].values[0]
        role_batter = 1.0 if role_guess == "batter" else 0.0
        role_bowler = 1.0 if role_guess == "bowler" else 0.0
        role_allrounder = 1.0 if role_guess == "allrounder" else 0.0
        role_flex = 1.0 if role_guess == "flex" else 0.0

        # Venue/opposition performance heuristics (season-agnostic aggregates)
        venue_perf = 0.0
        opp_perf = 0.0
        # These can be refined: for now simple deltas vs overall
        # If you have per-venue breakdown, compute SR/Avg at venue vs global

        # Engineered indexes (bounded z-style scalers with safe defaults)
        def safe_scale(x, a=0, b=1, cap=1.0):
            val = (x - a) / (b - a) if b > a else 0.0
            return float(np.clip(val, 0, cap))

        batting_form_index = safe_scale(bat_sr, a=90, b=160, cap=1.2) * 0.6 + safe_scale(bat_avg, a=20, b=50, cap=1.2) * 0.4
        bowling_form_index = safe_scale(1.0/(bowl_sr+1e-6), a=0.02, b=0.06, cap=1.2) * 0.5 + safe_scale(1.0/(bowl_eco+1e-6), a=0.08, b=0.14, cap=1.2) * 0.5
        allround_index = 0.5 * batting_form_index + 0.5 * bowling_form_index
        consistency_score = safe_scale((bat_runs + wkts*20)/max(1.0, (outs+overs)), a=0.2, b=2.0)
        role_strength = (role_batter*batting_form_index + role_bowler*bowling_form_index + role_allrounder*allround_index + role_flex*0.5)

        # Phase strengths (placeholders based on role and form)
        pp_strength = batting_form_index * 0.6 + (1.0/(bowl_eco+1e-6))*0.1
        middle_strength = batting_form_index * 0.5 + bowling_form_index * 0.5
        death_strength = batting_form_index * 0.4 + (1.0/(bowl_sr+1e-6))*0.2

        # Adjustments
        venue_adj = venue_perf * 0.2
        opp_adj = opp_perf * 0.2
        season_career_delta = ( (bat_sr - (c_bat_runs/max(1.0,c_bat_balls)*100.0 if c_bat_balls>0 else 0.0)) * 0.01 )

        return {
            "bat_runs": bat_runs, "bat_balls": bat_balls, "bat_sr": bat_sr, "bat_avg": bat_avg, "boundary_pct": boundary_pct,
            "wkts": wkts, "bowl_overs": overs, "bowl_eco": bowl_eco, "bowl_sr": bowl_sr, "dot_pct": dot_pct,
            "matches": max(1.0, outs + overs/4.0),  # proxy
            "role_batter": role_batter, "role_bowler": role_bowler, "role_allrounder": role_allrounder, "role_flex": role_flex,
            "venue_perf": venue_perf, "opp_perf": opp_perf,
            "batting_form_index": batting_form_index, "bowling_form_index": bowling_form_index, "allround_index": allround_index,
            "consistency_score": consistency_score, "role_strength": role_strength,
            "pp_strength": pp_strength, "middle_strength": middle_strength, "death_strength": death_strength,
            "venue_adj": venue_adj, "opp_adj": opp_adj, "season_career_delta": season_career_delta
        }
