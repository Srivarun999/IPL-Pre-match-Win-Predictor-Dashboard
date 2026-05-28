# IPL Pre-Match Win Predictor (Streamlit)

A production-ready Streamlit web app that predicts IPL match win probabilities using pre-match, player-driven features only. It generalizes player skill independent of franchise, includes venue and match context, and avoids recent team-form leakage. Time-aware season split is used for training.

## Features
- Pre-match inputs: season, teams, venue, match type, day/night.
- Season-wise squad picker and Playing XI selection.
- Player stats derived from historical ball-by-ball or match-level data.
- Engineered player indexes and aggregated team strengths.
- XGBoost / HistGradientBoosting / RandomForest candidates with calibrated XGBoost fallback for deployment.
- Time-aware train/validation/test by season.
- Explanation cards that highlight the strongest model-driven feature gaps for the winning probability.
- Flexible data loader that maps common IPL schemas.

## Current training details
- Training frame is built from unique match rows with season-aware team-form and player-strength aggregates.
- The model uses two perspectives per match (team1 vs team2 and team2 vs team1) to keep the probability estimate symmetric.
- Candidate models: XGBoost, HistGradientBoosting, RandomForest; the best validation log-loss model is kept and optionally calibrated.
- Final deployment uses the calibrated model when available, otherwise the best XGBoost model, then the Logistic Regression fallback.
- Current artifact summary (from `artifacts/metrics_report.json`): validation accuracy 0.95, ROC-AUC 0.9946, log loss 0.4106; test accuracy 0.6667, ROC-AUC 0.6391, log loss 0.6626.

## Project Structure
See repository tree in the root message. Key scripts:
- `train_model.py`: trains models and saves artifacts.
- `evaluate_model.py`: loads artifacts and evaluates on the test split.
- `app.py`: Streamlit UI for predictions using selected XIs.

## Data
Place your dataset in `data/` (e.g., `data/IPL.csv`). The loader supports flexible column names, including the provided wide schema. At minimum, your data should contain:
- date/year/season
- venue/city
- teams (batting_team / bowling_team or team1/team2)
- result/winner (for training labels)
- player involvement to infer squads and stats (batter/bowler, runs, wickets, balls, etc.)

If squads-by-season are missing, the app infers player pools from season match appearances.

## Install
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run locally
```bash
python train_model.py
python evaluate_model.py
streamlit run app.py
```

## Deployment notes
- The Streamlit app reads artifacts from `artifacts/` and shows interactive win probabilities, season success rate, and feature-importance explanations.
- Re-train after changing the dataset or the player-feature heuristics so the explanation cards and model weights stay aligned.
