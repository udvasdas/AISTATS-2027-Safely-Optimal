"""Stopping thresholds beta(t, delta)."""

import numpy as np
from scipy.special import lambertw

_ZETA2 = np.pi ** 2 / 6


def _h_inv(x):
    """Inverse of h(u) = u - log(u) on u >= 1 (x >= 1)."""
    return float(np.real(-lambertw(-np.exp(-x), k=-1)))


def _h_tilde(x, k=1.5):
    if x >= _h_inv(1.0 / np.log(k)):
        u = _h_inv(x)
        return np.exp(1.0 / u) * u
    return k * (x - np.log(np.log(k)))


def kk_T(x):
    """Calibration function T(x) of Kaufmann & Koolen (2021), T(x) ~ x + 4 log(1 + x + sqrt(2x))."""
    return 2.0 * _h_tilde((_h_inv(1.0 + x) + np.log(2.0 * _ZETA2)) / 2.0)


def beta_theory(N, delta, d):
    """Lemma 2: beta(t, delta) = 2(d+1) sum_a log(4 + log N_a) + K T(log(1/delta) / K)."""
    K = len(N)
    return 2.0 * (d + 1) * np.sum(np.log(4.0 + np.log(np.maximum(N, 1)))) + K * kk_T(np.log(1.0 / delta) / K)


def beta_stylized(t, delta):
    """log((1 + log t) / delta), the heuristic threshold used in the experiments of Lardy et al. (2025)."""
    return np.log((1.0 + np.log(t)) / delta)
