import numpy as np
from scipy.stats import rankdata

def pseudo_observations(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("frontier_invalid_asset_history")
    if not np.all(np.isfinite(values)):
        raise ValueError("frontier_nonfinite_asset_history")
    ranks = np.column_stack([rankdata(values[:, col], method="average") for col in range(values.shape[1])])
    return np.clip(ranks / float(values.shape[0] + 1), 1e-8, 1.0 - 1e-8).astype(np.float64)
