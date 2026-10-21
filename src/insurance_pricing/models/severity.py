"""Claim severity: a constant or gamma GLM, with a flat loading for large losses."""

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor

from insurance_pricing.features.rating_factors import SEVERITY_FEATURES


def large_loss_loading(df: pd.DataFrame) -> float:
    """Ratio of total claim cost to total capped claim cost, at least 1.

    Multiplying a model of capped (attritional) severity by this ratio adds back
    the cost above the large-loss threshold, spread evenly over all claims.
    """
    return float(df["ClaimAmount"].sum() / df["AttritionalAmount"].sum())


class SeverityModel:
    """Expected cost per claim, constant or a gamma GLM with log link.

    The model is fitted to ``AvgAttritionalAmount`` (claims capped at the
    large-loss threshold) with ``ClaimNb`` as sample weight, on policies with at
    least one claim. This keeps single large losses from distorting the
    relativities. The cost above the threshold is added back with a flat
    ``loading`` estimated on the training data, so predictions are expected
    full cost per claim. Predictions are defined for every policy, with or
    without claims.

    With no ``features`` (the default, see ``SEVERITY_FEATURES``) the capped
    severity is the claim-weighted mean, which is what an intercept-only gamma
    GLM estimates. With features, the first level of each categorical rating
    factor is its reference level, so coefficients are log relativities against
    that level.
    """

    def __init__(self, features: list[str] = SEVERITY_FEATURES, alpha: float = 0.0):
        self.features = features
        self.glm = (
            GeneralizedLinearRegressor(family="gamma", alpha=alpha, drop_first=True)
            if features
            else None
        )
        self.mean = 0.0
        self.loading = 1.0

    def fit(self, df: pd.DataFrame) -> "SeverityModel":
        claims = df[df["ClaimNb"] > 0]
        if self.glm is None:
            self.mean = float(
                claims["AttritionalAmount"].sum() / claims["ClaimNb"].sum()
            )
        else:
            self.glm.fit(
                claims[self.features],
                claims["AvgAttritionalAmount"],
                sample_weight=claims["ClaimNb"],
            )
        self.loading = large_loss_loading(claims)
        return self

    def predict_severity(self, df: pd.DataFrame) -> np.ndarray:
        """Expected cost per claim, including the large-loss loading."""
        if self.glm is None:
            return np.full(len(df), self.mean * self.loading)
        severity: np.ndarray = self.glm.predict(df[self.features]) * self.loading
        return severity

    def relativities(self) -> pd.DataFrame:
        """Multiplicative effect of each coefficient on severity, with base cost.

        The intercept includes the large-loss loading, so its relativity is the
        base expected cost per claim. A constant model has only the intercept.
        """
        if self.glm is None:
            coef = pd.Series([np.log(self.mean * self.loading)], index=["intercept"])
        else:
            coef = pd.Series(
                [self.glm.intercept_ + np.log(self.loading), *self.glm.coef_],
                index=["intercept", *self.glm.feature_names_],
            )
        return pd.DataFrame({"coef": coef, "relativity": np.exp(coef)})
