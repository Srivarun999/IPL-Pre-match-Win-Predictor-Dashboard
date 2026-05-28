from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import pandas as pd

TEAM_FEATURES = [
    "bat_runs","bat_balls","bat_sr","bat_avg","boundary_pct",
    "wkts","bowl_overs","bowl_eco","bowl_sr","dot_pct",
    "matches","batting_form_index","bowling_form_index","allround_index",
    "consistency_score","role_strength","pp_strength","middle_strength","death_strength",
    "venue_adj","opp_adj","season_career_delta",
    "role_batter","role_bowler","role_allrounder","role_flex"
]

def _agg(players: List[Dict[str, float]]) -> Dict[str, float]:
    if not players:
        return {k:0.0 for k in TEAM_FEATURES}
    arr = {k: np.array([p.get(k, 0.0) for p in players], dtype=float) for k in TEAM_FEATURES}
    # Averages with some sums
    out = {}
    for k, v in arr.items():
        if k in ["bat_runs", "wkts", "bowl_overs"]:  # sums can show volume
            out[f"team_{k}_sum"] = float(np.nansum(v))
            out[f"team_{k}_avg"] = float(np.nanmean(v))
        else:
            out[f"team_{k}_avg"] = float(np.nanmean(v))
    # Specialized strengths
    out["team_batting_strength"] = out.get("team_batting_form_index_avg", 0.0) * 0.7 + out.get("team_bat_sr_avg", 0.0)/200.0 * 0.3
    out["team_bowling_strength"] = out.get("team_bowling_form_index_avg", 0.0) * 0.7 + (1.0/(out.get("team_bowl_eco_avg", 0.0)+1e-6))*0.05
    out["team_allround_strength"] = out.get("team_allround_index_avg", 0.0)
    out["team_top_order_strength"] = out.get("team_bat_avg_avg", 0.0) * 0.6 + out.get("team_pp_strength_avg", 0.0) * 0.4
    out["team_middle_order_strength"] = out.get("team_middle_strength_avg", 0.0)
    out["team_death_overs_strength"] = out.get("team_death_strength_avg", 0.0)
    out["team_powerplay_bowling_strength"] = (1.0/(out.get("team_bowl_sr_avg", 0.0)+1e-6)) * 0.1 + (1.0/(out.get("team_bowl_eco_avg", 0.0)+1e-6))*0.05
    out["team_spin_strength"] = out.get("team_role_bowler_avg", 0.0) * 0.3  # placeholder
    out["team_pace_strength"] = out.get("team_role_bowler_avg", 0.0) * 0.3  # placeholder
    # Balance
    out["team_balance_score"] = (out["team_batting_strength"] + out["team_bowling_strength"] + out["team_allround_strength"])/3.0
    return out

def aggregate_team_features(player_feature_bundle: Dict[str, Any]) -> Dict[str, float]:
    players = player_feature_bundle.get("_players", [])
    return _agg(players)


def compute_current_season_form_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """Compute current-season team form features using only prior matches in the same season."""
    d = match_df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    required = {"season", "team1", "team2", "venue", "winner"}
    if not required.issubset(set(d.columns)):
        return d

    d = d.dropna(subset=["season", "team1", "team2", "venue", "winner"]).copy()
    d = d.sort_values(["season", "date", "match_id"], kind="mergesort") if "match_id" in d.columns else d.sort_values(["season", "date"], kind="mergesort")
    d = d.reset_index(drop=True)

    histories: Dict[str, List[Dict[str, Any]]] = {}
    out_rows = []

    for _, row in d.iterrows():
        season = int(row["season"])
        team1 = str(row["team1"])
        team2 = str(row["team2"])
        venue = str(row["venue"])
        winner = str(row["winner"])

        def stats_for(team: str, opp: str, venue_name: str) -> Dict[str, float]:
            hist = histories.get(team, [])
            games = len(hist)
            wins = sum(1 for x in hist if x["winner"] == team)
            recent = hist[-5:]
            recent_wins = sum(1 for x in recent if x["winner"] == team)
            venue_hist = [x for x in hist if x["venue"] == venue_name]
            venue_games = len(venue_hist)
            venue_wins = sum(1 for x in venue_hist if x["winner"] == team)
            opp_hist = [x for x in hist if x["opp"] == opp]
            opp_games = len(opp_hist)
            opp_wins = sum(1 for x in opp_hist if x["winner"] == team)
            return {
                f"{team.lower().replace(' ', '_')}_season_games": float(games),
                f"{team.lower().replace(' ', '_')}_season_win_rate": float(wins / games) if games else 0.0,
                f"{team.lower().replace(' ', '_')}_recent_form_5": float(recent_wins / min(5, len(recent))) if recent else 0.0,
                f"{team.lower().replace(' ', '_')}_venue_win_rate": float(venue_wins / venue_games) if venue_games else 0.0,
                f"{team.lower().replace(' ', '_')}_vs_opp_win_rate": float(opp_wins / opp_games) if opp_games else 0.0,
                f"{team.lower().replace(' ', '_')}_season_wins": float(wins),
            }

        form1 = stats_for(team1, team2, venue)
        form2 = stats_for(team2, team1, venue)

        out_rows.append({
            **row.to_dict(),
            **form1,
            **form2,
            "team1_form_gap": form1.get(f"{team1.lower().replace(' ', '_')}_season_win_rate", 0.0) - form2.get(f"{team2.lower().replace(' ', '_')}_season_win_rate", 0.0),
            "team1_recent_form_gap": form1.get(f"{team1.lower().replace(' ', '_')}_recent_form_5", 0.0) - form2.get(f"{team2.lower().replace(' ', '_')}_recent_form_5", 0.0),
        })

        histories.setdefault(team1, []).append({"winner": winner == team1, "venue": venue, "opp": team2})
        histories.setdefault(team2, []).append({"winner": winner == team2, "venue": venue, "opp": team1})

    return pd.DataFrame(out_rows)

def build_matchup_features(team1_feats: Dict[str, float], team2_feats: Dict[str, float], match_type: str, day_or_night: str) -> Dict[str, float]:
    playoff_pressure = 0.0
    if match_type in ["Qualifier", "Eliminator", "Final"]:
        playoff_pressure = 0.1
    night_adv = 0.05 if day_or_night == "Night" else 0.0

    return {
        "team_batting_vs_bowling": team1_feats.get("team_batting_strength",0) - team2_feats.get("team_bowling_strength",0),
        "opp_batting_vs_bowling": team2_feats.get("team_batting_strength",0) - team1_feats.get("team_bowling_strength",0),
        "team_venue_fit": team1_feats.get("team_pp_strength_avg",0) + team1_feats.get("team_death_strength_avg",0),
        "opp_venue_fit": team2_feats.get("team_pp_strength_avg",0) + team2_feats.get("team_death_strength_avg",0),
        "playoff_pressure_score": playoff_pressure,
        "night_match_advantage": night_adv
    }
