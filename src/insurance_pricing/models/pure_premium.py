"""Pure premium as expected frequency times expected severity."""

from typing import Protocol, Self

import numpy as np
import pandas as pd

from insurance_pricing.models.frequency import FrequencyModel
from insurance_pricing.models.severity import SeverityModel


class FrequencyPredictor(Protocol):
    def fit(self, df: pd.DataFrame) -> Self: ...
    def predict_frequency(self, df: pd.DataFrame) -> np.ndarray: ...


class SeverityPredictor(Protocol):
    def fit(self, df: pd.DataFrame) -> Self: ...
    def predict_severity(self, df: pd.DataFrame) -> np.ndarray: ...


class PurePremiumModel:
    """Expected claim cost per year of exposure: frequency x severity.

    Assumes the number of claims and the cost per claim are independent given
    the rating factors, so the expected cost is the product of their means.
    Defaults to the GLMs; any frequency and severity models with the same
    interface, such as the GBMs, can be passed in instead.
    """

    def __init__(
        self,
        frequency: FrequencyPredictor | None = None,
        severity: SeverityPredictor | None = None,
    ):
        self.frequency = frequency if frequency is not None else FrequencyModel()
        self.severity = severity if severity is not None else SeverityModel()

    def fit(self, df: pd.DataFrame) -> "PurePremiumModel":
        self.frequency.fit(df)
        self.severity.fit(df)
        return self

    def predict_pure_premium(self, df: pd.DataFrame) -> np.ndarray:
        """Expected claim cost per year of exposure."""
        frequency = self.frequency.predict_frequency(df)
        severity = self.severity.predict_severity(df)
        pure_premium: np.ndarray = frequency * severity
        return pure_premium

    def predict_cost(self, df: pd.DataFrame) -> np.ndarray:
        """Expected claim cost over each policy's exposure."""
        cost: np.ndarray = self.predict_pure_premium(df) * df["Exposure"].to_numpy()
        return cost

    def relativities(self) -> pd.DataFrame:
        """Frequency, severity and combined relativity for each coefficient.

        Only defined when both components are GLMs. A coefficient in only one of
        the two models has relativity 1 in the other. The intercept row holds
        the base frequency, base severity and their product, the base pure
        premium.
        """
        if not isinstance(self.frequency, FrequencyModel) or not isinstance(
            self.severity, SeverityModel
        ):
            raise TypeError("Relativities need GLM frequency and severity models.")
        table = pd.concat(
            {
                "frequency": self.frequency.relativities()["relativity"],
                "severity": self.severity.relativities()["relativity"],
            },
            axis=1,
        ).fillna(1.0)
        table["pure_premium"] = table["frequency"] * table["severity"]
        return table
