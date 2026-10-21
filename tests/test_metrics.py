import numpy as np
import pytest

from insurance_pricing.evaluation.metrics import (
    POISSON,
    calibration_table,
    d2_score,
    evaluate,
    gini,
)


@pytest.fixture
def simulated() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed frequency, true frequency and exposure for 100k policies."""
    rng = np.random.default_rng(0)
    true_frequency = rng.gamma(shape=2.0, scale=0.05, size=100_000)
    exposure = rng.uniform(0.1, 1.0, size=100_000)
    claims = rng.poisson(true_frequency * exposure)
    return claims / exposure, true_frequency, exposure


def test_d2_is_zero_for_weighted_mean_and_one_for_perfect_fit() -> None:
    y = np.array([0.0, 1.0, 2.0, 1.0])
    weight = np.array([1.0, 2.0, 1.0, 4.0])
    mean = np.average(y, weights=weight)

    assert d2_score(y, np.full(4, mean), weight, POISSON) == pytest.approx(0.0)
    assert d2_score(y, y + 1e-12, weight, POISSON) == pytest.approx(1.0)


def test_gini_rewards_ranking_not_level(
    simulated: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y, true_frequency, exposure = simulated
    random = np.random.default_rng(1).permutation(true_frequency)

    informed = gini(y, true_frequency, exposure)

    assert informed > 0.2
    assert gini(y, random, exposure) == pytest.approx(0.0, abs=0.05)
    # Any increasing transform of the prediction ranks the same.
    assert gini(y, 3 * true_frequency**2, exposure) == pytest.approx(informed)


def test_calibration_table_bands_have_equal_weight(
    simulated: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y, true_frequency, exposure = simulated

    table = calibration_table(y, true_frequency, exposure, n_bins=5)

    assert table.index.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(table["weight"], exposure.sum() / 5, rtol=0.01)
    assert table["predicted"].is_monotonic_increasing
    np.testing.assert_allclose(table["ratio"], 1.0, atol=0.1)


def test_evaluate_reports_balance(
    simulated: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y, _, exposure = simulated
    mean = np.average(y, weights=exposure)

    metrics = evaluate(y, np.full(len(y), 2 * mean), exposure, POISSON)

    assert metrics["balance"] == pytest.approx(2.0)
    assert metrics["d2"] < 0
    assert set(metrics) == {"deviance", "d2", "gini", "balance"}
