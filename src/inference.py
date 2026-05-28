from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import pandas as pd
import joblib

def load_artifacts(artifacts_dir: Path | str) -> Dict[str, Any]:
    p = Path(artifacts_dir)
    artifacts: Dict[str, Any] = {}
    if (p/"model_xgb.pkl").exists():
        artifacts["xgb"] = joblib.load(p/"model_xgb.pkl")
    if (p/"model_lr.pkl").exists():
        artifacts["lr"] = joblib.load(p/"model_lr.pkl")
    if (p/"calibrator.pkl").exists():
        artifacts["calibrator"] = joblib.load(p/"calibrator.pkl")
    if (p/"feature_columns.json").exists():
        with open(p/"feature_columns.json","r") as f:
            artifacts["feature_cols"] = json.load(f)
    if (p/"feature_importance.csv").exists():
        artifacts["feature_importance"] = pd.read_csv(p/"feature_importance.csv")
    return artifacts

def align_features(df: pd.DataFrame, feature_cols: Optional[list]) -> pd.DataFrame:
    if feature_cols is None:
        return df
    X = pd.DataFrame(columns=feature_cols)
    for c in feature_cols:
        X[c] = df[c] if c in df.columns else 0.0
    return X

def predict_proba_with_models(X_raw: pd.DataFrame, artifacts: Dict[str, Any]) -> tuple[pd.Series, str]:
    feature_cols = artifacts.get("feature_cols")
    X = align_features(X_raw, feature_cols)
    model = artifacts.get("calibrator") or artifacts.get("xgb") or artifacts.get("lr")
    if model is None:
        # Fallback constant
        preds = pd.Series([0.5]*len(X))
        return preds, "none"
    proba = model.predict_proba(X.values)[:,1]
    return pd.Series(proba), ("calibrated_xgb" if artifacts.get("calibrator") else "xgb" if artifacts.get("xgb") else "lr")

def get_feature_importance(artifacts: Dict[str, Any]) -> Optional[pd.DataFrame]:
    return artifacts.get("feature_importance")
