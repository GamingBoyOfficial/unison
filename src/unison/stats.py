"""Small-sample statistics (no SciPy needed)."""

from __future__ import annotations

import numpy as np

# two-sided 95% critical values of Student's t, by degrees of freedom
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
         30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_crit(df: int) -> float:
    """95% critical value. Between table entries, the larger (safer) value is used."""
    if df < 1:
        return float("nan")
    if df > 120:
        return 1.96
    return _T975[max(k for k in _T975 if k <= df)]


def mean_ci(values) -> tuple[float, float]:
    """Mean and the half-width of its 95% confidence interval (t-based)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(t_crit(n - 1) * v.std(ddof=1) / np.sqrt(n))
