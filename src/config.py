from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    data_file: Path = Path("data/IPL.csv")
    artifacts_dir: Path = Path("artifacts")

@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    xgb_n_estimators: int = 500
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 5
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 1.0
    xgb_reg_alpha: float = 0.0
    calibrate: bool = True
