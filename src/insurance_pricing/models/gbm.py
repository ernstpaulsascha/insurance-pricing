"""Gradient boosted trees for frequency and severity, as a benchmark for the GLMs.

They have the same interface as the GLM models, so ``PurePremiumModel`` can
combine either. Trees find non-linear effects and interactions on the raw
features themselves, which shows how much structure the GLMs miss.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from insurance_pricing.data.split import train_test_split_policies
from insurance_pricing.features.rating_factors import GBM_FEATURES
from insurance_pricing.models.severity import large_loss_loading

# Share of the training data held out to decide when to stop adding trees.
VALIDATION_SIZE = 0.1
DEFAULT_PARAMS: dict[str, Any] = {
    "max_iter": 2000,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "early_stopping": True,
    # Validation comes from a group split in ``fit``, not a random row sample.
    "validation_fraction": None,
    "n_iter_no_change": 50,
}


def _fit_with_early_stopping(
    model: HistGradientBoostingRegressor,
    df: pd.DataFrame,
    features: list[str],
    target: str,
    weight: str,
    random_state: int,
) -> None:
    train, validation = train_test_split_policies(
        df, test_size=VALIDATION_SIZE, random_state=random_state
    )
    model.fit(
        train[features],
        train[target],
        sample_weight=train[weight],
        X_val=validation[features],
        y_val=validation[target],
        sample_weight_val=validation[weight],
    )


class GBMFrequencyModel:
    """Gradient boosted trees with Poisson loss predicting claims per year.

    Fits ``Frequency`` with ``Exposure`` as sample weight, like the GLM.
    """

    def __init__(
        self,
        *,
        features: list[str] = GBM_FEATURES,
        random_state: int = 42,
        **params: Any,
    ):
        self.features = features
        self.random_state = random_state
        self.gbm = HistGradientBoostingRegressor(
            loss="poisson", random_state=random_state, **(DEFAULT_PARAMS | params)
        )

    def fit(self, df: pd.DataFrame) -> "GBMFrequencyModel":
        _fit_with_early_stopping(
            self.gbm, df, self.features, "Frequency", "Exposure", self.random_state
        )
        return self

    def predict_frequency(self, df: pd.DataFrame) -> np.ndarray:
        """Expected claims per year of exposure."""
        frequency: np.ndarray = self.gbm.predict(df[self.features])
        return frequency

    def predict_claims(self, df: pd.DataFrame) -> np.ndarray:
        """Expected number of claims over each policy's exposure."""
        claims: np.ndarray = self.predict_frequency(df) * df["Exposure"].to_numpy()
        return claims


class GBMSeverityModel:
    """Gradient boosted trees with gamma loss predicting the cost per claim.

    Like the GLM, it fits capped amounts (``AvgAttritionalAmount``) weighted by
    ``ClaimNb`` on policies with claims, and adds back the large-loss
    ``loading``.
    """

    def __init__(
        self,
        *,
        features: list[str] = GBM_FEATURES,
        random_state: int = 42,
        **params: Any,
    ):
        self.features = features
        self.random_state = random_state
        self.gbm = HistGradientBoostingRegressor(
            loss="gamma", random_state=random_state, **(DEFAULT_PARAMS | params)
        )
        self.loading = 1.0

    def fit(self, df: pd.DataFrame) -> "GBMSeverityModel":
        claims = df[df["ClaimNb"] > 0]
        _fit_with_early_stopping(
            self.gbm,
            claims,
            self.features,
            "AvgAttritionalAmount",
            "ClaimNb",
            self.random_state,
        )
        self.loading = large_loss_loading(claims)
        return self

    def predict_severity(self, df: pd.DataFrame) -> np.ndarray:
        """Expected cost per claim, including the large-loss loading."""
        severity: np.ndarray = self.gbm.predict(df[self.features]) * self.loading
        return severity
