import numpy as np
import pandas as pd
import pytest

from insurance_pricing.models.frequency import FrequencyModel

BASE_FREQUENCY = 0.1
DIESEL_RELATIVITY = 1.5


@pytest.fixture
def simulated_policies(simulated_features: pd.DataFrame) -> pd.DataFrame:
    """Policies whose claims depend only on fuel type, with known relativity."""
    df = simulated_features.copy()
    diesel = (df["VehGas"] == "Diesel").to_numpy()
    frequency = BASE_FREQUENCY * np.where(diesel, DIESEL_RELATIVITY, 1.0)
    df["ClaimNb"] = np.random.default_rng(1).poisson(frequency * df["Exposure"])
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    return df


def test_recovers_known_relativity(simulated_policies: pd.DataFrame) -> None:
    model = FrequencyModel().fit(simulated_policies)

    relativities = model.relativities()["relativity"]

    assert relativities["VehGas[Diesel]"] == pytest.approx(DIESEL_RELATIVITY, rel=0.1)


def test_predicted_claims_balance_observed_on_training_data(
    simulated_policies: pd.DataFrame,
) -> None:
    model = FrequencyModel().fit(simulated_policies)

    predicted = model.predict_claims(simulated_policies).sum()

    # Exact at the optimum; glum's default solver tolerance leaves ~0.04% here.
    assert predicted == pytest.approx(simulated_policies["ClaimNb"].sum(), rel=1e-3)


def test_predict_claims_scales_frequency_by_exposure(
    simulated_policies: pd.DataFrame,
) -> None:
    model = FrequencyModel().fit(simulated_policies)

    frequency = model.predict_frequency(simulated_policies)
    claims = model.predict_claims(simulated_policies)

    np.testing.assert_allclose(claims, frequency * simulated_policies["Exposure"])
    assert (frequency > 0).all()
