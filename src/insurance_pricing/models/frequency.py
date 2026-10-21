"""Poisson GLM for claim frequency."""

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor

from insurance_pricing.features.rating_factors import GLM_FEATURES


class FrequencyModel:
    """Poisson GLM with log link predicting claims per year of exposure.

    Fits ``Frequency = ClaimNb / Exposure`` with ``Exposure`` as sample weight,
    which is equivalent to fitting ``ClaimNb`` with a ``log(Exposure)`` offset.

    By default the GLM is unpenalised, as is usual for a tariff. The first level
    of each categorical rating factor is its reference level, so coefficients
    are log relativities against that level.
    """

    def __init__(self, features: list[str] = GLM_FEATURES, alpha: float = 0.0):
        self.features = features
        self.glm = GeneralizedLinearRegressor(
            family="poisson", alpha=alpha, drop_first=True
        )

    def fit(self, df: pd.DataFrame) -> "FrequencyModel":
        self.glm.fit(df[self.features], df["Frequency"], sample_weight=df["Exposure"])
        return self

    def predict_frequency(self, df: pd.DataFrame) -> np.ndarray:
        """Expected claims per year of exposure."""
        frequency: np.ndarray = self.glm.predict(df[self.features])
        return frequency

    def predict_claims(self, df: pd.DataFrame) -> np.ndarray:
        """Expected number of claims over each policy's exposure."""
        claims: np.ndarray = self.predict_frequency(df) * df["Exposure"].to_numpy()
        return claims

    def relativities(self) -> pd.DataFrame:
        """Multiplicative effect of each coefficient on frequency, with base rate."""
        coef = pd.Series(
            [self.glm.intercept_, *self.glm.coef_],
            index=["intercept", *self.glm.feature_names_],
        )
        return pd.DataFrame({"coef": coef, "relativity": np.exp(coef)})
