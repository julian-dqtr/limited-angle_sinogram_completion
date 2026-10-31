"""Paired statistics of the evaluation: two models scored on the same test samples.

The test samples are resampled in pairs, so the difficulty of each sample cancels out. These tests compare two
trained networks; a claim about an architecture or a loss also needs the seed replicas (results/model_analysis.ipynb).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import wilcoxon

N_BOOT = 10_000


def paired_wilcoxon(differences) -> float:
    """Two-sided p-value of the paired Wilcoxon signed-rank test (1 if every difference is 0)."""
    differences = np.asarray(differences, dtype=float)
    return float(wilcoxon(differences).pvalue) if np.any(differences != 0) else 1.0


def paired_bootstrap(candidate, reference, relative: bool = True, n_boot: int = N_BOOT,
                     rng: Optional[np.random.Generator] = None) -> dict:
    """Effect on the mean of a metric, with a paired bootstrap 95 % interval (percentile) and the Wilcoxon p-value.

    relative=True: mean(candidate) / mean(reference) - 1, otherwise mean(candidate) - mean(reference). Without
    ``rng`` every call draws from the seed 0, so an interval does not depend on the order of the comparisons.
    """
    candidate, reference = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError(f"Expected two paired 1-D arrays, got {candidate.shape} and {reference.shape}")
    rng = np.random.default_rng(0) if rng is None else rng

    def effect(c: np.ndarray, r: np.ndarray):
        return c.mean(axis=-1) / r.mean(axis=-1) - 1.0 if relative else c.mean(axis=-1) - r.mean(axis=-1)

    draws = rng.integers(0, len(reference), size=(n_boot, len(reference)))
    low, high = np.percentile(effect(candidate[draws], reference[draws]), [2.5, 97.5])
    return {"effect": float(effect(candidate, reference)), "ci_low": float(low), "ci_high": float(high),
            "p_value": paired_wilcoxon(candidate - reference)}


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error rate), in the input order; NaN stays NaN.

    Reject the hypothesis i at level alpha iff its adjusted p-value is below alpha.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return adjusted
    order = np.argsort(p[valid])
    stepped = np.minimum(np.maximum.accumulate(p[valid][order] * (m - np.arange(m))), 1.0)
    ranked = np.empty(m)
    ranked[order] = stepped
    adjusted[valid] = ranked
    return adjusted
