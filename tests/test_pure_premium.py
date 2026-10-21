import numpy as np
import pandas as pd
import pytest
from conftest import DIESEL_FREQUENCY_RELATIVITY, DIESEL_SEVERITY_RELATIVITY

from insurance_pricing.features.rating_factors import GLM_FEATURES
from insurance_pricing.models.pure_premium import PurePremiumModel
from insurance_pricing.models.severity import SeverityModel


def test_pure_premium_is_frequency_times_severity(
    simulated_claims: pd.DataFrame,
) -> None:
    model = PurePremiumModel().fit(simulated_claims)

    pure_premium = model.predict_pure_premium(simulated_claims)

    np.testing.assert_allclose(
        pure_premium,
        model.frequency.predict_frequency(simulated_claims)
        * model.severity.predict_severity(simulated_claims),
    )
    np.testing.assert_allclose(
        model.predict_cost(simulated_claims),
        pure_premium * simulated_claims["Exposure"],
    )


def test_relativities_combine_frequency_and_severity(
    simulated_claims: pd.DataFrame,
) -> None:
    model = PurePremiumModel(severity=SeverityModel(features=GLM_FEATURES))
    model.fit(simulated_claims)

    diesel = model.relativities().loc["VehGas[Diesel]"]

    assert diesel["pure_premium"] == pytest.approx(
        diesel["frequency"] * diesel["severity"]
    )
    assert diesel["pure_premium"] == pytest.approx(
        DIESEL_FREQUENCY_RELATIVITY * DIESEL_SEVERITY_RELATIVITY, rel=0.1
    )


def test_constant_severity_leaves_only_frequency_relativities(
    simulated_claims: pd.DataFrame,
) -> None:
    model = PurePremiumModel().fit(simulated_claims)

    table = model.relativities().drop(index="intercept")

    assert (table["severity"] == 1.0).all()
    np.testing.assert_allclose(table["pure_premium"], table["frequency"])
