"""Metrics for comparing pricing models.

Every function takes the observed target, the model's prediction of it, and
the weight the model was fitted with:

- frequency: ``Frequency`` weighted by ``Exposure`` (Tweedie power 1, Poisson)
- severity: ``AvgClaimAmount`` weighted by ``ClaimNb`` (power 2, gamma)
- pure premium: ``PurePremium`` weighted by ``Exposure`` (power between 1 and 2,
  ``TWEEDIE``)
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import d2_tweedie_score, mean_tweedie_deviance

POISSON = 1.0
GAMMA = 2.0
# Compound Poisson-gamma; mostly zeros with a continuous positive tail.
TWEEDIE = 1.9

type Array = npt.ArrayLike


def deviance(y: Array, pred: Array, weight: Array, power: float) -> float:
    """Weighted mean Tweedie deviance; lower is better."""
    return float(mean_tweedie_deviance(y, pred, sample_weight=weight, power=power))


def d2_score(y: Array, pred: Array, weight: Array, power: float) -> float:
    """Share of deviance explained against predicting the weighted mean of ``y``."""
    return float(d2_tweedie_score(y, pred, sample_weight=weight, power=power))


def lorenz_curve(
    y: Array, pred: Array, weight: Array
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Ordered Lorenz curve: cumulative shares of weight and of observed losses.

    Policies are sorted from lowest to highest prediction. Returns the
    cumulative share of weight and of observed losses (``y * weight``), both
    starting at 0 and ending at 1.
    """
    y, pred, weight = _as_arrays(y, pred, weight)
    order = np.argsort(pred, kind="stable")
    weight_share = np.concatenate([[0.0], np.cumsum(weight[order])])
    loss_share = np.concatenate([[0.0], np.cumsum((y * weight)[order])])
    return weight_share / weight_share[-1], loss_share / loss_share[-1]


def gini(y: Array, pred: Array, weight: Array) -> float:
    """Gini coefficient of the ordered Lorenz curve; how well ``pred`` ranks risks.

    Twice the area between the diagonal and the ``lorenz_curve``. Ranking
    perfectly gives the highest possible value, random ranking gives about 0.
    Only the order of ``pred`` matters, not its level.
    """
    weight_share, loss_share = lorenz_curve(y, pred, weight)
    return float(1 - 2 * np.trapezoid(loss_share, weight_share))


def calibration_table(
    y: Array, pred: Array, weight: Array, n_bins: int = 10
) -> pd.DataFrame:
    """Compare observed and predicted values within bands of predicted risk.

    Bands hold roughly equal weight, from lowest predicted risk (band 0) to
    highest. A well calibrated model has ``ratio`` (observed / predicted) close
    to 1 in every band.
    """
    y, pred, weight = _as_arrays(y, pred, weight)
    order = np.argsort(pred, kind="stable")
    sorted_weight = weight[order]
    # Place each policy by the weight share at its midpoint.
    position = (np.cumsum(sorted_weight) - sorted_weight / 2) / weight.sum()
    bands = np.empty(len(pred), dtype=int)
    bands[order] = np.minimum((position * n_bins).astype(int), n_bins - 1)

    df = pd.DataFrame(
        {
            "band": bands,
            "weight": weight,
            "observed": y * weight,
            "predicted": pred * weight,
        }
    )
    table = df.groupby("band").sum()
    table["observed"] /= table["weight"]
    table["predicted"] /= table["weight"]
    table["ratio"] = table["observed"] / table["predicted"]
    return table


def evaluate(y: Array, pred: Array, weight: Array, power: float) -> dict[str, float]:
    """Headline metrics for one model on one dataset."""
    y, pred, weight = _as_arrays(y, pred, weight)
    return {
        "deviance": deviance(y, pred, weight, power),
        "d2": d2_score(y, pred, weight, power),
        "gini": gini(y, pred, weight),
        # Overall predicted / observed; 1 means the total is right.
        "balance": float((pred * weight).sum() / (y * weight).sum()),
    }


def _as_arrays(
    y: Array, pred: Array, weight: Array
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return (
        np.asarray(y, dtype=float),
        np.asarray(pred, dtype=float),
        np.asarray(weight, dtype=float),
    )
