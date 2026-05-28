import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np

from src.config import AppConfig, ModelConfig
from src.data_loader import SeasonCatalog, load_matches_flexible
from src.preprocessing import FlexibleColumnMapper
from src.player_features import PlayerStatsService
from src.feature_engineering import build_match_context_features, build_selected_xi_player_features
from src.team_features import aggregate_team_features, build_matchup_features
from src.inference import load_artifacts, predict_proba_with_models, get_feature_importance
from src.utils import set_seed


st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")
set_seed(42)

@st.cache_resource
def bootstrap() -> Tuple[pd.DataFrame, FlexibleColumnMapper, SeasonCatalog, PlayerStatsService, Dict[str, Any]]:
    cfg = AppConfig()
    data_path = cfg.data_file
    raw = load_matches_flexible(data_path)
    mapper = FlexibleColumnMapper.infer(raw)
    df = mapper.remap(raw).copy()

    season_catalog = SeasonCatalog(df)
    player_service = PlayerStatsService(df)
    artifacts = load_artifacts(cfg.artifacts_dir)
    return df, mapper, season_catalog, player_service, artifacts

def slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def season_success_rate(df: pd.DataFrame, team: str, season: int) -> float:
    sub = df[(df["season"] == season) & ((df["team1"] == team) | (df["team2"] == team))].copy()
    if sub.empty:
        return float("nan")
    wins = (sub["winner"] == team).sum()
    return wins / len(sub)

def season_team_form_features(df: pd.DataFrame, season: int, team: str, opp: str, venue: str) -> Dict[str, float]:
    sub = df[(df["season"] == season) & ((df["team1"] == team) | (df["team2"] == team))].copy()
    if sub.empty:
        return {f"{slug(team)}_season_games": 0.0, f"{slug(team)}_season_win_rate": 0.0, f"{slug(team)}_recent_form_5": 0.0,
                f"{slug(team)}_venue_win_rate": 0.0, f"{slug(team)}_vs_opp_win_rate": 0.0, f"{slug(team)}_season_wins": 0.0}
    sub = sub.drop_duplicates(subset=["match_id"] if "match_id" in sub.columns else ["season", "team1", "team2", "venue", "date"], keep="first")
    sub = sub.sort_values(["date"] if "date" in sub.columns else ["season"], kind="mergesort")
    wins = int((sub["winner"] == team).sum())
    recent = sub.tail(5)
    recent_wins = int((recent["winner"] == team).sum())
    venue_hist = sub[sub["venue"] == venue]
    venue_wins = int((venue_hist["winner"] == team).sum())
    opp_hist = sub[(sub["team1"] == opp) | (sub["team2"] == opp)]
    opp_wins = int((opp_hist["winner"] == team).sum())
    return {
        f"{slug(team)}_season_games": float(len(sub)),
        f"{slug(team)}_season_win_rate": float(wins / len(sub)) if len(sub) else 0.0,
        f"{slug(team)}_recent_form_5": float(recent_wins / len(recent)) if len(recent) else 0.0,
        f"{slug(team)}_venue_win_rate": float(venue_wins / len(venue_hist)) if len(venue_hist) else 0.0,
        f"{slug(team)}_vs_opp_win_rate": float(opp_wins / len(opp_hist)) if len(opp_hist) else 0.0,
        f"{slug(team)}_season_wins": float(wins),
    }


def readable_feature_label(feature_name: str) -> str:
    label = feature_name.replace("team_", "").replace("opp_", "opposition ").replace("_", " ")
    label = label.replace("batting", "batting").replace("bowling", "bowling")
    return label.strip().capitalize()


def explain_prediction(team1: str, team2: str, row_t1: Dict[str, float], row_t2: Dict[str, float], win_prob_t1: float, win_prob_t2: float, feature_importance: Any = None) -> List[str]:
    reasons = []
    edge = abs(win_prob_t1 - win_prob_t2)
    leader = team1 if win_prob_t1 >= win_prob_t2 else team2
    trailer = team2 if leader == team1 else team1
    if edge < 0.05:
        reasons.append("The model sees this as a close contest; the probability split is very narrow, so the edge is not a certainty.")
    else:
        reasons.append(f"The model gives {leader} a {edge * 100:.1f}-point edge, driven by the strongest pre-match signals in this matchup.")

    candidate_features = []
    if feature_importance is not None and not feature_importance.empty:
        for feature in feature_importance["feature"].tolist():
            if feature in row_t1 and feature in row_t2 and feature not in {"label", "match_id"}:
                diff = float(row_t1.get(feature, 0.0)) - float(row_t2.get(feature, 0.0))
                if abs(diff) > 1e-6:
                    candidate_features.append((feature, abs(diff), diff, float(feature_importance.loc[feature_importance["feature"] == feature, "importance"].iloc[0])))

    candidate_features.sort(key=lambda item: (item[3], item[1]), reverse=True)
    for feature, _, diff, _ in candidate_features[:4]:
        if diff > 0:
            reasons.append(f"{leader} has the stronger {readable_feature_label(feature)} signal in this setup, which matches the model’s probability tilt toward {leader}.")
        else:
            reasons.append(f"{trailer} has the stronger {readable_feature_label(feature)} signal here, so the model keeps the contest tighter than the headline probability suggests.")

    if not candidate_features:
        reasons.append("The model is relying on team balance, venue fit, and player-strength aggregates rather than recent streaks.")

    if len(reasons) < 4:
        reasons.append("This probability is a pre-match estimate built from season context, venue context, and selected XI strength, not a post-match result.")

    return reasons[:5]


def render_sidebar(season_catalog: SeasonCatalog) -> Dict[str, Any]:
    st.sidebar.header("Match Setup")

    season = st.sidebar.selectbox("Season year", options=season_catalog.seasons_sorted())
    teams = season_catalog.teams_in_season(season)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", options=teams, key="team1_sidebar")
    with col2:
        team2 = st.selectbox("Team 2", options=[t for t in teams if t != team1], key="team2_sidebar")

    venues = season_catalog.venues_in_season(season)
    venue = st.sidebar.selectbox("Venue", options=sorted(venues))
    match_type = st.sidebar.selectbox("Match type", options=["Normal", "Qualifier", "Eliminator", "Final"])
    day_or_night = st.sidebar.selectbox("Match time", options=["Day", "Night"])

    return {
        "season": season,
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "match_type": match_type,
        "day_or_night": day_or_night,
    }

def render_xi_selectors(player_service: PlayerStatsService, season: int, team1: str, team2: str) -> Tuple[List[str], List[str]]:
    st.subheader("Playing XIs")
    t1_pool = player_service.players_for_team_in_season(team1, season)
    t2_pool = player_service.players_for_team_in_season(team2, season)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<span style='color:#38bdf8;font-weight:700'>{team1} Squad (Season {season})</span>", unsafe_allow_html=True)
        t1_xi = []
        for player in t1_pool:
            if st.checkbox(player, key=f"t1_{slug(player)}"):
                t1_xi.append(player)
        st.caption(f"Selected: {len(t1_xi)}")
    with col2:
        st.markdown(f"<span style='color:#f472b6;font-weight:700'>{team2} Squad (Season {season})</span>", unsafe_allow_html=True)
        t2_xi = []
        for player in t2_pool:
            if st.checkbox(player, key=f"t2_{slug(player)}"):
                t2_xi.append(player)
        st.caption(f"Selected: {len(t2_xi)}")

    return t1_xi, t2_xi

def main():
    st.markdown("""
    <style>
    :root { color-scheme: dark; }
    .stApp { background: linear-gradient(135deg, #04101f 0%, #102a43 42%, #1d4ed8 100%); color: #f8fbff; }
    .stTitle, .stSubheader, h1, h2, h3, h4, p, label, .stMarkdown, .stCaption, .stTextInput label, .stSelectbox label, .stMultiSelect label, .stCheckbox label, .stRadio label { color: #f8fbff !important; }
    .stMetric > div > div > div { color: #ffffff !important; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div, .stMultiSelect > div > div, .stNumberInput > div > div > input {
        background-color: #07111f !important;
        color: #f8fbff !important;
        border: 1px solid #60a5fa !important;
        border-radius: 8px;
    }
    .stSelectbox div[role="option"], .stMultiSelect div[role="option"] { color: #07111f !important; }
    .stButton > button {
        background: linear-gradient(135deg, #22c55e, #3b82f6);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.25);
    }
    .stButton > button:hover { filter: brightness(1.08); }
    div[data-testid="stAlert"], div[data-testid="stNotification"] {
        background: rgba(8, 15, 30, 0.92);
        border: 1px solid rgba(191, 219, 254, 0.35);
        color: #eff6ff;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #102a43 45%, #172554 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.25);
    }
    [data-testid="stSidebar"] * { color: #eff6ff !important; }
    [data-testid="stSidebar"] .stSelectbox > div, [data-testid="stSidebar"] .stMultiSelect > div {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(191, 219, 254, 0.25);
        color: #eff6ff;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"] { color: #e2e8f0; background: rgba(15, 23, 42, 0.65); border-radius: 10px; }
    .stTabs [aria-selected="true"] { background: rgba(30, 64, 175, 0.95); color: #eff6ff; }
    </style>
    """, unsafe_allow_html=True)

    df, mapper, season_catalog, player_service, artifacts = bootstrap()
    st.title("IPL Pre-Match Win Predictor")
    st.caption("Pre-match, player-driven, venue-aware probabilities. Trained with season-aware splits. No team recent-form leakage.")
    st.sidebar.info("Tip: switch the season, venue, or XI to compare how the probability changes in real time.")

    inputs = render_sidebar(season_catalog)
    team1, team2 = inputs["team1"], inputs["team2"]

    t1_xi, t2_xi = render_xi_selectors(player_service, inputs["season"], team1, team2)

    col = st.container()
    with col:
        st.markdown("---")
        disabled = len(t1_xi) != 11 or len(t2_xi) != 11
        if disabled:
            st.warning("Please select exactly 11 players for both teams.")
        if st.button("Predict Win Probabilities", disabled=disabled, type="primary"):
            # Build feature rows for both team perspectives: (team1 vs team2) and (team2 vs team1)
            ctx_row_t1 = build_match_context_features(
                season=inputs["season"],
                venue=inputs["venue"],
                city=season_catalog.city_for_venue(inputs["venue"]),
                match_type=inputs["match_type"],
                day_or_night=inputs["day_or_night"],
                team1=team1, team2=team2
            )
            ctx_row_t2 = build_match_context_features(
                season=inputs["season"],
                venue=inputs["venue"],
                city=season_catalog.city_for_venue(inputs["venue"]),
                match_type=inputs["match_type"],
                day_or_night=inputs["day_or_night"],
                team1=team2, team2=team1
            )

            # Player-level features for selected XIs (pre-match only, career till season-1 and current-season to date assumptions)
            t1_players = build_selected_xi_player_features(player_service, inputs["season"], team1, t1_xi, opp=team2, venue=inputs["venue"])
            t2_players = build_selected_xi_player_features(player_service, inputs["season"], team2, t2_xi, opp=team1, venue=inputs["venue"])

            # Team aggregations
            t1_team_feats = aggregate_team_features(t1_players)
            t2_team_feats = aggregate_team_features(t2_players)

            # Matchup engineered features
            matchup_t1 = build_matchup_features(t1_team_feats, t2_team_feats, inputs["match_type"], inputs["day_or_night"])
            matchup_t2 = build_matchup_features(t2_team_feats, t1_team_feats, inputs["match_type"], inputs["day_or_night"])

            # Assemble model rows
            form_t1 = season_team_form_features(df, inputs["season"], team1, team2, inputs["venue"])
            form_t2 = season_team_form_features(df, inputs["season"], team2, team1, inputs["venue"])
            row_t1 = {**ctx_row_t1, **t1_team_feats, **matchup_t1, **form_t1,
                      "team1_form_gap": form_t1.get(f"{slug(team1)}_season_win_rate", 0.0) - form_t2.get(f"{slug(team2)}_season_win_rate", 0.0)}
            row_t2 = {**ctx_row_t2, **t2_team_feats, **matchup_t2, **form_t2,
                      "team2_form_gap": form_t2.get(f"{slug(team2)}_season_win_rate", 0.0) - form_t1.get(f"{slug(team1)}_season_win_rate", 0.0)}
            X_pred = pd.DataFrame([row_t1, row_t2])

            # Predict
            model_loaded = any(artifacts.get(key) is not None for key in ["calibrator", "xgb", "lr"])
            if not model_loaded:
                st.error("No trained model artifacts found. Run `python train_model.py` to build the model artifacts before using prediction.")
                st.info("If you have trained the model already, place the files in the `artifacts/` directory: model_xgb.pkl, model_lr.pkl, calibrator.pkl, feature_columns.json.")
                return

            preds, used_model = predict_proba_with_models(X_pred, artifacts)
            p_t1 = float(preds.iloc[0])
            p_t2 = float(preds.iloc[1])

            # Season success rates
            s1 = season_success_rate(df, team1, inputs["season"])
            s2 = season_success_rate(df, team2, inputs["season"])

            # Display results
            tabs = st.tabs(["Prediction", "Feature signals", "How to read it"])

            with tabs[0]:
                st.markdown("### Result")
                if p_t1 > p_t2:
                    st.success(f"Predicted winner: {team1}")
                elif p_t2 > p_t1:
                    st.success(f"Predicted winner: {team2}")
                else:
                    st.info("Too close to call.")

                st.metric(f"{team1} win %", f"{p_t1*100:.1f}%")
                st.metric(f"{team2} win %", f"{p_t2*100:.1f}%")
                st.progress(min(1.0, max(p_t1, p_t2)), text=f"Confidence edge: {abs(p_t1 - p_t2) * 100:.1f}%")
                st.caption(f"{team1} season {inputs['season']} success rate: {s1*100:.1f}%" if pd.notna(s1) else f"{team1}: N/A")
                st.caption(f"{team2} season {inputs['season']} success rate: {s2*100:.1f}%" if pd.notna(s2) else f"{team2}: N/A")
                st.caption(f"Model used: {used_model}")

            with tabs[1]:
                st.markdown("### Key Features (Global Importance)")
                imp = get_feature_importance(artifacts)
                if imp is not None and not imp.empty:
                    top_imp = imp.head(12)
                    st.bar_chart(top_imp.set_index("feature")["importance"])
                    with st.expander("What these signals mean"):
                        st.write("Higher bars are the strongest model signals. The app now uses these weights to explain why the probability is leaning one way.")
                else:
                    st.write("Importance not available for the active model.")

            with tabs[2]:
                st.markdown("### Why this prediction?")
                for reason in explain_prediction(team1, team2, row_t1, row_t2, p_t1, p_t2, get_feature_importance(artifacts)):
                    st.write(f"- {reason}")
                st.info("This is a pre-match estimate built from season context, selected XI strength, venue fit, and matchup balance — not a guaranteed result.")

            st.markdown("---")
            st.markdown("### Notes")
            st.write("- Probabilities are pre-match and player-driven, independent of recent team streaks.")
            st.write("- Squad lists are inferred from season match sheets if explicit squads are missing.")
            st.write("- Ensure your dataset columns are mapped correctly via the flexible loader.")

if __name__ == "__main__":
    main()
