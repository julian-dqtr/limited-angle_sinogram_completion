"""Statistics behind the angular-reach and oversmoothing figures and the paired comparisons (no model, no GPU)."""
import numpy as np
import pytest

from src.metrics import PerViewMoments, smoothness_metrics
from src.statistics import holm, paired_bootstrap, paired_wilcoxon


def accumulate(samples) -> PerViewMoments:
    moments = PerViewMoments()
    for sample in samples:
        moments.update(sample)
    return moments


def test_per_view_variance_is_the_population_variance_averaged_over_the_detector():
    rng = np.random.default_rng(0)
    samples = 3.0 + rng.normal(size=(50, 7, 5)) * rng.uniform(0.1, 2.0, size=(1, 7, 5))
    np.testing.assert_allclose(accumulate(samples).per_view_variance(), samples.var(axis=0).mean(axis=1), rtol=1e-10)


def test_identical_outputs_have_exactly_zero_variance():
    """Beyond its reach a graph model outputs the same values for every sample: the variance must be exactly 0."""
    constant = 0.1 + 1e3 * np.random.default_rng(1).normal(size=(7, 5))
    assert np.all(accumulate([constant] * 200).per_view_variance() == 0.0)


def test_ground_truth_variance_is_the_floor_of_any_prediction_independent_of_the_data():
    rng = np.random.default_rng(2)
    targets = rng.normal(size=(30, 7, 5))
    floor = accumulate(targets).per_view_variance()
    for constant in (np.zeros((7, 5)), rng.normal(size=(7, 5)), targets.mean(axis=0)):
        per_view_mse = ((targets - constant) ** 2).mean(axis=(0, 2))
        assert np.all(per_view_mse >= floor - 1e-12)
    # The bound is reached by the mean of the samples.
    np.testing.assert_allclose(((targets - targets.mean(axis=0)) ** 2).mean(axis=(0, 2)), floor)


# Two blocks of missing views around an acquired window, as in the real geometry.
MISSING = np.array([True] * 4 + [False] * 3 + [True] * 4)


def test_smoothness_ratios_are_one_for_the_ground_truth():
    target = np.random.default_rng(3).normal(size=(11, 16))
    assert smoothness_metrics(target, target, MISSING) == pytest.approx(
        {"dirichlet_angle_ratio": 1.0, "dirichlet_detector_ratio": 1.0})


def test_a_washed_out_wedge_has_no_angular_energy():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(11, 16))
    washed_out = target.copy()
    washed_out[MISSING] = target[MISSING].mean(axis=0)  # the same row in every missing view
    metrics = smoothness_metrics(washed_out, target, MISSING)
    assert metrics["dirichlet_angle_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["dirichlet_detector_ratio"] < 1.0


def test_smoothness_ignores_the_acquired_views():
    """Neither the acquired views nor the pairs of views across the window border enter the wedge indicators."""
    rng = np.random.default_rng(5)
    target, pred = rng.normal(size=(11, 16)), rng.normal(size=(11, 16))
    altered = pred.copy()
    altered[~MISSING] = 1e3 * rng.normal(size=((~MISSING).sum(), 16))
    assert smoothness_metrics(altered, target, MISSING) == pytest.approx(smoothness_metrics(pred, target, MISSING))


def test_holm_matches_the_step_down_procedure():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_holm_keeps_nan_out_of_the_family_and_caps_at_one():
    adjusted = holm([0.5, np.nan, 0.4])
    assert np.isnan(adjusted[1])
    np.testing.assert_allclose(adjusted[[0, 2]], [0.8, 0.8])
    assert holm([0.9, 0.8])[0] == 1.0


def test_paired_bootstrap_interval_contains_the_effect_and_is_reproducible():
    rng = np.random.default_rng(6)
    reference = rng.uniform(1.0, 2.0, size=200)
    candidate = reference * (0.9 + 0.02 * rng.normal(size=200))
    result = paired_bootstrap(candidate, reference)
    assert result["ci_low"] < result["effect"] < result["ci_high"] < 0.0
    assert result["effect"] == pytest.approx(candidate.mean() / reference.mean() - 1.0)
    assert paired_bootstrap(candidate, reference) == result
    absolute = paired_bootstrap(candidate, reference, relative=False)
    assert absolute["effect"] == pytest.approx(candidate.mean() - reference.mean())


def test_identical_models_have_no_effect_and_p_one():
    values = np.random.default_rng(7).uniform(size=50)
    result = paired_bootstrap(values, values)
    assert result["effect"] == result["ci_low"] == result["ci_high"] == 0.0
    assert result["p_value"] == paired_wilcoxon(np.zeros(50)) == 1.0
