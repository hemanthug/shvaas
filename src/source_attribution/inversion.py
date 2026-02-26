from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet

from .config import AttributionConfig


def infer_sources_elasticnet(
    A: np.ndarray,
    y: np.ndarray,
    config: AttributionConfig,
) -> tuple[np.ndarray, dict]:
    """Infer non-negative sparse source weights using ElasticNet."""
    if A.size == 0 or y.size == 0:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0), {
            "n_rows": int(A.shape[0] if A.ndim == 2 else 0),
            "n_sources": int(A.shape[1] if A.ndim == 2 else 0),
            "r2": np.nan,
            "mse": np.nan,
            "nonzero_coef": 0,
        }

    col_scale = np.std(A, axis=0)
    col_scale = np.where(col_scale > 1e-12, col_scale, 1.0)
    A_std = A / col_scale

    model = ElasticNet(
        alpha=config.elasticnet_alpha,
        l1_ratio=config.elasticnet_l1_ratio,
        fit_intercept=False,
        positive=True,
        random_state=config.random_state,
        max_iter=5000,
    )
    model.fit(A_std, y)

    coef = model.coef_ / col_scale
    y_hat = A @ coef
    mse = float(np.mean((y - y_hat) ** 2))
    var_y = float(np.var(y))
    r2 = float(1.0 - mse / var_y) if var_y > 0 else np.nan

    diagnostics = {
        "n_rows": int(A.shape[0]),
        "n_sources": int(A.shape[1]),
        "r2": r2,
        "mse": mse,
        "nonzero_coef": int(np.count_nonzero(coef > 1e-12)),
    }
    return coef, diagnostics

