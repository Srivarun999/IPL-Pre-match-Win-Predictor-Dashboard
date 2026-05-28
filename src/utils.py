from __future__ import annotations
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def safe_log1p(x: float) -> float:
    import math
    if x <= -1:
        return 0.0
    return float(math.log1p(x))
