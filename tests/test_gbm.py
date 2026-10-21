from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import DIESEL_FREQUENCY_RELATIVITY, DIESEL_SEVERITY_RELATIVITY

from insurance_pricing.models.gbm import GBMFrequencyModel, GBMSeverityModel
from insurance_pricing.models.pure_premium import PurePremiumModel

# Fewer, faster trees than the defaults keep the tests quick.
FAST: dict[str, Any] = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 4}


def _diesel_ratio(predictions: np.ndarray, df: pd.DataFrame) -> float:
    diesel = (df["VehGas"] == "Diesel").to_numpy()
    return float(predictions[diesel].mean() / predictions[~diesel].mean())


def test_frequency_model_finds_diesel_effect(simulated_claims: pd.DataFrame) -> None:
    model = GBMFrequencyModel(**FAST).fit(simulated_claims)

    frequency = model.predict_frequency(simulated_claims)

    assert _diesel_ratio(frequency, simulated_claims) == pytest.approx(
        DIESEL_FREQUENCY_RELATIVITY, rel=0.1
    )
    np.testing.assert_allclose(
        model.predict_claims(simulated_claims),
        frequency * simulated_claims["Exposure"],
    )


def test_severity_model_finds_diesel_effect_and_applies_loading(
    simulated_claims: pd.DataFrame,
) -> None:
    with_large_loss = simulated_claims.copy()
    excess = 0.1 * simulated_claims["AttritionalAmount"].sum()
    policy = simulated_claims.index[simulated_claims["ClaimNb"] > 0][0]
    with_large_loss.loc[policy, "ClaimAmount"] += excess

    model = GBMSeverityModel(**FAST).fit(with_large_loss)
    severity = model.predict_severity(simulated_claims)

    assert model.loading == pytest.approx(1.1)
    assert _diesel_ratio(severity, simulated_claims) == pytest.approx(
        DIESEL_SEVERITY_RELATIVITY, rel=0.1
    )
    np.testing.assert_allclose(
        severity, 1.1 * model.gbm.predict(simulated_claims[model.features])
    )


def test_early_stopping_ends_before_all_trees(simulated_claims: pd.DataFrame) -> None:
    # The only signal is fuel type, so it stops well before the 2000 tree limit.
    model = GBMFrequencyModel(learning_rate=0.3).fit(simulated_claims)

    assert model.gbm.n_iter_ < model.gbm.max_iter


def test_pure_premium_combines_gbms(simulated_claims: pd.DataFrame) -> None:
    model = PurePremiumModel(GBMFrequencyModel(**FAST), GBMSeverityModel(**FAST))

    model.fit(simulated_claims)

    np.testing.assert_allclose(
        model.predict_pure_premium(simulated_claims),
        model.frequency.predict_frequency(simulated_claims)
        * model.severity.predict_severity(simulated_claims),
    )
    with pytest.raises(TypeError):
        model.relativities()
