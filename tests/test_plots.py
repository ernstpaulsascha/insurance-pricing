from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from insurance_pricing.evaluation.metrics import (  # noqa: E402
    calibration_table,
    lorenz_curve,
)
from insurance_pricing.evaluation.plots import (  # noqa: E402
    plot_actual_vs_expected,
    plot_calibration,
    plot_lorenz,
)

type Simulated = tuple[np.ndarray, np.ndarray, np.ndarray]


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture
def simulated() -> Simulated:
    """Observed frequency, true frequency and exposure for 10k policies."""
    rng = np.random.default_rng(0)
    true_frequency = rng.gamma(shape=2.0, scale=0.05, size=10_000)
    exposure = rng.uniform(0.1, 1.0, size=10_000)
    claims = rng.poisson(true_frequency * exposure)
    return claims / exposure, true_frequency, exposure


def _lines(ax: Axes) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        str(line.get_label()): (
            np.asarray(line.get_xdata(), dtype=float),
            np.asarray(line.get_ydata(), dtype=float),
        )
        for line in ax.get_lines()
    }


def test_lorenz_plots_each_model_and_diagonal(simulated: Simulated) -> None:
    y, true_frequency, exposure = simulated

    ax = plot_lorenz(y, exposure, {"model": true_frequency})

    lines = _lines(ax)
    model = next(name for name in lines if name.startswith("model (Gini "))
    assert "random" in lines
    assert not any(name.startswith("oracle") for name in lines)
    expected = lorenz_curve(y, true_frequency, exposure)
    np.testing.assert_allclose(lines[model][0], expected[0])
    np.testing.assert_allclose(lines[model][1], expected[1])


def test_lorenz_oracle_is_opt_in(simulated: Simulated) -> None:
    y, true_frequency, exposure = simulated

    ax = plot_lorenz(y, exposure, {"model": true_frequency}, oracle=True)

    assert any(name.startswith("oracle (Gini ") for name in _lines(ax))


def test_calibration_plots_band_means(simulated: Simulated) -> None:
    y, true_frequency, exposure = simulated

    ax = plot_calibration(y, exposure, {"model": true_frequency}, n_bins=5)

    x, observed = _lines(ax)["model"]
    table = calibration_table(y, true_frequency, exposure, n_bins=5)
    np.testing.assert_allclose(x, table["predicted"])
    np.testing.assert_allclose(observed, table["observed"])


def test_actual_vs_expected_weights_means_per_level() -> None:
    factor = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 0.0, 2.0, 4.0])
    weight = np.array([1.0, 3.0, 1.0, 1.0])
    pred = np.array([0.5, 0.5, 3.0, 3.0])

    ax = plot_actual_vs_expected(factor, y, weight, {"model": pred})

    lines = _lines(ax)
    # Level a: (1*1 + 0*3) / 4; level b: (2 + 4) / 2.
    np.testing.assert_allclose(lines["observed"][1], [0.25, 3.0])
    np.testing.assert_allclose(lines["model"][1], [0.5, 3.0])
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["a", "b"]
    bars = ax.figure.axes[1].patches
    assert [bar.get_height() for bar in bars if isinstance(bar, Rectangle)] == [
        4.0,
        2.0,
    ]


def test_plots_draw_on_given_axes(simulated: Simulated) -> None:
    y, true_frequency, exposure = simulated
    _, (left, right) = plt.subplots(1, 2)

    assert plot_lorenz(y, exposure, {"m": true_frequency}, ax=left) is left
    assert plot_calibration(y, exposure, {"m": true_frequency}, ax=right) is right
