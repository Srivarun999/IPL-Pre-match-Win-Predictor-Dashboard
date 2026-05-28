import argparse
from pathlib import Path
from src.model_training import training_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/IPL.csv")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--valid_season", type=int, default=2019)
    ap.add_argument("--test_season", type=int, default=2020)
    args = ap.parse_args()

    report = training_pipeline(Path(args.data), Path(args.artifacts), args.valid_season, args.test_season)
    print("Training report:")
    print(report)

if __name__ == "__main__":
    main()
