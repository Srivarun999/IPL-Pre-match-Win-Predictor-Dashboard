from __future__ import annotations
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import xgboost as xgb

from .preprocessing import FlexibleColumnMapper, basic_clean
from .data_loader import load_matches_flexible
from .player_features import PlayerStatsService
from .feature_engineering import build_match_context_features
from .team_features import aggregate_team_features, build_matchup_features, compute_current_season_form_features

@dataclass
class SplitPlan:
    train_seasons: List[int]
    valid_season: int
    test_season: int

def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Construct training rows per match from actual played XIs if available; often, ball-by-ball identifies participants.
    # Here we approximate team XI strengths by inferring primary players appearing in that match's season team.
    # For every match row, we need: season, team1, team2, venue, match_type/daynight (if missing, default).
    d = basic_clean(df).copy()
    # Ensure essential columns
    for c in ["season","team1","team2","venue"]:
        if c not in d.columns:
            raise ValueError(f"Missing required column: {c}")

    # The raw input is often ball-by-ball data. Build one training example per unique match,
    # not per delivery, otherwise training explodes in time and memory.
    if "match_id" in d.columns:
        match_key = ["match_id"]
    else:
        match_key = ["season", "team1", "team2", "venue", "date"]

    d = d.drop_duplicates(subset=match_key, keep="first").copy()

    # Add current-season team form features (prior matches only in same season)
    d = compute_current_season_form_features(d)

    # Winner/label
    if "winner" not in d.columns:
        # try to derive from win_outcome or match_won_by + team name; here assume 'winner' exists ideally
        raise ValueError("Winner column is required for training.")
    d = d.dropna(subset=["team1","team2","venue","season","winner"])
    d["day_or_night"] = d.get("day_or_night", pd.Series(["Day"]*len(d)))
    d["match_type"] = d.get("match_type", pd.Series(["Normal"]*len(d)))

    # Player service for aggregates
    ps = PlayerStatsService(d)

    rows = []
    for idx, row in d.iterrows():
        season = int(row["season"])
        t1, t2 = str(row["team1"]), str(row["team2"])
        venue = str(row["venue"])
        mt = str(row["match_type"])
        dn = str(row["day_or_night"])
        # Infer squads from season teams; choose top 11 by appearances heuristic
        t1_pool = ps.players_for_team_in_season(t1, season)[:11]
        t2_pool = ps.players_for_team_in_season(t2, season)[:11]
        if len(t1_pool) == 0 or len(t2_pool) == 0:
            # skip if we can't infer players
            continue

        t1_players = {"_players":[ps.player_vector(p, season, opp=t2, venue=venue) for p in t1_pool]}
        t2_players = {"_players":[ps.player_vector(p, season, opp=t1, venue=venue) for p in t2_pool]}
        t1_team = aggregate_team_features(t1_players)
        t2_team = aggregate_team_features(t2_players)
        matchup_t1 = build_matchup_features(t1_team, t2_team, mt, dn)

        ctx_t1 = build_match_context_features(season, venue, None, mt, dn, t1, t2)
        ctx_t2 = build_match_context_features(season, venue, None, mt, dn, t2, t1)

        toss_win = 1.0 if str(row.get("toss_winner", "")).strip() == t1 else 0.0
        toss_decision = 1.0 if str(row.get("toss_decision", "")).lower() in {"bat", "batting"} else 0.0
        team_runs = float(row.get("team_runs", 0.0) or 0.0)
        team_wkts = float(row.get("team_wicket", 0.0) or 0.0)

        # Two rows: perspective team1 and team2 with binary label
        y1 = 1 if str(row["winner"]) == t1 else 0
        y2 = 1 if str(row["winner"]) == t2 else 0
        form_row = {
            k: v for k, v in row.to_dict().items()
            if k.endswith("_season_games") or k.endswith("_season_win_rate") or k.endswith("_recent_form_5")
            or k.endswith("_venue_win_rate") or k.endswith("_vs_opp_win_rate") or k.endswith("_season_wins")
            or k in {"team1_form_gap", "team1_recent_form_gap"}
        }
        rows.append({**ctx_t1, **t1_team, **matchup_t1, **form_row,
                     "toss_win_flag": toss_win, "toss_decision_enc": toss_decision,
                     "match_runs_scored": team_runs, "match_wickets_lost": team_wkts,
                     "label": y1})
        rows.append({**ctx_t2, **t2_team, **matchup_t1, **form_row,
                     "toss_win_flag": 1.0 - toss_win, "toss_decision_enc": 1.0 - toss_decision,
                     "match_runs_scored": float(row.get("team_runs", 0.0) or 0.0), "match_wickets_lost": float(row.get("team_wicket", 0.0) or 0.0),
                     "label": y2})

    train_df = pd.DataFrame(rows)
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train_df

def fit_models(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], cfg_dir: Path, calibrate: bool = True) -> Dict[str, Any]:
    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_val, y_val = val_df[feature_cols].values, val_df["label"].values

    # Try a small set of high-signal models and pick the one with the best validation log-loss.
    candidate_models = {
        "xgb": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
            tree_method="hist",
            verbosity=0,
        ),
        "hgb": HistGradientBoostingClassifier(random_state=42, learning_rate=0.05, max_depth=6, max_iter=300),
        "rf": RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1, max_depth=10),
    }

    best_model = None
    best_score = None
    best_name = None
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, proba)
        if best_score is None or score < best_score:
            best_score = score
            best_name = name
            best_model = model

    xgb_model = best_model

    calibrator = None
    if calibrate:
        calibrator = CalibratedClassifierCV(xgb_model, cv=3, method="isotonic")
        calibrator.fit(X_val, y_val)

    lr = LogisticRegression(max_iter=400, n_jobs=1)
    lr.fit(X_train, y_train)

    # Save artifacts
    Path(cfg_dir).mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(xgb_model, Path(cfg_dir)/"model_xgb.pkl")
    if lr is not None:
        joblib.dump(lr, Path(cfg_dir)/"model_lr.pkl")
    if calibrator is not None:
        joblib.dump(calibrator, Path(cfg_dir)/"calibrator.pkl")
    with open(Path(cfg_dir)/"feature_columns.json","w") as f:
        json.dump(feature_cols, f, indent=2)
    return {
        "xgb": xgb_model,
        "lr": lr,
        "calibrator": calibrator,
        "feature_cols": feature_cols
    }

def evaluate_split(df_eval: pd.DataFrame, model, feature_cols: List[str]) -> Dict[str, Any]:
    X = df_eval[feature_cols].values
    y = df_eval["label"].values
    proba = model.predict_proba(X)[:,1]
    y_pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:
        auc = float("nan")
    try:
        ll = log_loss(y, proba)
    except ValueError:
        ll = float("nan")

    cm = confusion_matrix(y, y_pred).tolist()
    return {"accuracy": acc, "roc_auc": auc, "log_loss": ll, "confusion_matrix": cm}

def training_pipeline(data_path: Path, artifacts_dir: Path, valid_season: int, test_season: int) -> Dict[str, Any]:
    raw = load_matches_flexible(data_path)
    mapper = FlexibleColumnMapper.infer(raw)
    df = mapper.remap(raw)

    train_all = build_training_frame(df)
    if train_all.empty:
        raise ValueError("Training frame is empty. Check data and column mappings.")

    # Time-aware split: train < valid < test
    available_seasons = sorted(pd.unique(train_all["season"]).tolist())
    valid_season = valid_season if valid_season in available_seasons else available_seasons[-2]
    test_season = test_season if test_season in available_seasons else available_seasons[-1]

    train_df = train_all[train_all["season"] < valid_season]
    val_df = train_all[train_all["season"] == valid_season]
    test_df = train_all[train_all["season"] == test_season]

    # Features
    feature_cols = [c for c in train_all.columns if c not in ["label"]]

    models = fit_models(train_df, val_df, feature_cols, artifacts_dir, calibrate=True)

    # Choose calibrated XGB if exists else XGB else LR
    active = models["calibrator"] if models.get("calibrator") is not None else models["xgb"]

    # Eval
    report = {
        "train": evaluate_split(train_df, active, feature_cols),
        "valid": evaluate_split(val_df, active, feature_cols),
        "test": evaluate_split(test_df, active, feature_cols),
    }

    # Save metrics and feature importance
    import json
    with open(Path(artifacts_dir)/"metrics_report.json","w") as f:
        json.dump(report, f, indent=2)

    # Feature importance for XGB
    try:
        import joblib
        xgb_model = models["xgb"]
        importances = xgb_model.feature_importances_
        fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
        fi.to_csv(Path(artifacts_dir)/"feature_importance.csv", index=False)
    except Exception:
        pass

    # Save mapper for reference
    with open(Path(artifacts_dir)/"feature_columns.json","w") as f:
        json.dump(feature_cols, f, indent=2)

    return report
