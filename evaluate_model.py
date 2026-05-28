import argparse
from pathlib import Path
import json
import pandas as pd

from src.data_loader import load_matches_flexible
from src.preprocessing import FlexibleColumnMapper, basic_clean
from src.model_training import build_training_frame, evaluate_split
from src.inference import load_artifacts, align_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/IPL.csv")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--test_season", type=int, default=2020)
    args = ap.parse_args()

    raw = load_matches_flexible(args.data)
    mapper = FlexibleColumnMapper.infer(raw)
    df = mapper.remap(raw)
    frame = build_training_frame(df)
    test_df = frame[frame["season"] == args.test_season]
    arts = load_artifacts(args.artifacts)
    feature_cols = arts.get("feature_cols")
    model = arts.get("calibrator") or arts.get("xgb") or arts.get("lr")
    if model is None or feature_cols is None:
        raise RuntimeError("Artifacts missing. Train first.")

    report = evaluate_split(test_df, model, feature_cols)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
