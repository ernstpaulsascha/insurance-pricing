"""Diagnostic plots for comparing pricing models.

Like the metrics, every plot takes the observed target and weight, plus one or
more models' predictions keyed by model name. Each function draws on ``ax`` (a
new figure if omitted) and returns it, so plots can be combined into figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

from insurance_pricing.evaluation.metrics import (
    calibration_table,
    gini,
    lorenz_curve,
)

type Array = npt.ArrayLike


def _axes(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_lorenz(
    y: Array,
    weight: Array,
    predictions: dict[str, Array],
    oracle: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Ordered Lorenz curves, with Gini coefficients in the legend.

    The diagonal is random ranking. A curve further below the diagonal ranks
    risks better. With ``oracle``, also draws the curve from ranking policies by
    their observed outcome. With mostly zero claims that bound is far out of any
    model's reach, so it is off by default.
    """
    ax = _axes(ax)
    curves = {"oracle": y, **predictions} if oracle else predictions
    for name, pred in curves.items():
        weight_share, loss_share = lorenz_curve(y, pred, weight)
        is_oracle = oracle and name == "oracle"
        ax.plot(
            weight_share,
            loss_share,
            label=f"{name} (Gini {gini(y, pred, weight):.3f})",
            linestyle="--" if is_oracle else "-",
            color="grey" if is_oracle else None,
        )
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, label="random")
    ax.set_xlabel("Cumulative share of exposure, lowest predicted risk first")
    ax.set_ylabel("Cumulative share of observed losses")
    ax.set_title("Lorenz curve")
    ax.legend()
    return ax


def plot_calibration(
    y: Array,
    weight: Array,
    predictions: dict[str, Array],
    n_bins: int = 10,
    ax: Axes | None = None,
) -> Axes:
    """Observed against predicted mean per band of predicted risk.

    Bands hold equal weight (see ``calibration_table``). A well calibrated model
    lies on the diagonal in every band, not only in total. Both axes are
    logarithmic, so the many low-risk bands are not squeezed together by the
    few high-risk ones.
    """
    ax = _axes(ax)
    lower, upper = np.inf, 0.0
    for name, pred in predictions.items():
        table = calibration_table(y, pred, weight, n_bins=n_bins)
        ax.plot(table["predicted"], table["observed"], marker="o", label=name)
        means = table[["predicted", "observed"]].to_numpy()
        lower, upper = min(lower, means.min()), max(upper, means.max())
    ax.plot(
        [lower, upper], [lower, upper], color="black", linewidth=0.8, label="perfect"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        # Plain numbers instead of powers of ten on the ticks.
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(ScalarFormatter())
    ax.set_xlabel("Predicted mean per band")
    ax.set_ylabel("Observed mean per band")
    ax.set_title(f"Calibration by {n_bins} bands of predicted risk")
    ax.legend()
    return ax


def plot_actual_vs_expected(
    factor: Array,
    y: Array,
    weight: Array,
    predictions: dict[str, Array],
    ax: Axes | None = None,
) -> Axes:
    """Observed and predicted mean per level of one rating factor.

    Bars show the weight in each level (right axis), so sparse levels can be
    told apart from real misfit. Bin a continuous factor before passing it in.
    """
    ax = _axes(ax)
    y_arr = np.asarray(y, dtype=float)
    weight_arr = np.asarray(weight, dtype=float)
    columns = {"weight": weight_arr, "observed": y_arr * weight_arr}
    columns |= {
        name: np.asarray(pred, dtype=float) * weight_arr
        for name, pred in predictions.items()
    }
    table = pd.DataFrame(columns).groupby(np.asarray(factor), observed=True).sum()
    means = table.drop(columns="weight").div(table["weight"], axis=0)
    positions = np.arange(len(table))

    bars = ax.twinx()
    bars.bar(positions, table["weight"], color="lightgrey")
    bars.set_ylabel("Weight")
    # Draw the lines above the bars.
    ax.set_zorder(bars.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.plot(positions, means["observed"], marker="o", color="black", label="observed")
    for name in predictions:
        ax.plot(positions, means[name], marker="o", label=name)
    ax.set_xticks(positions, [str(level) for level in table.index], rotation=45)
    ax.set_ylabel("Mean per level")
    ax.legend()
    return ax
