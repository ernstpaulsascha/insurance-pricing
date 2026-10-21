import numpy as np
import pandas as pd
import pytest

from insurance_pricing.features.rating_factors import GLM_FEATURES, SEVERITY_FEATURES
from insurance_pricing.models.severity import SeverityModel

BOTH = pytest.mark.parametrize(
    "features", [SEVERITY_FEATURES, GLM_FEATURES], ids=["constant", "glm"]
)

BASE_SEVERITY = 1000.0
DIESEL_RELATIVITY = 1.3
GAMMA_SHAPE = 2.0


@pytest.fixture
def simulated_policies(simulated_features: pd.DataFrame) -> pd.DataFrame:
    """Policies whose claim cost depends only on fuel type, with known relativity.

    Policies without claims have no average amounts (NaN), as in real data.
    There are no large losses, so capped and full amounts are equal.
    """
    rng = np.random.default_rng(1)
    df = simulated_features.copy()
    diesel = (df["VehGas"] == "Diesel").to_numpy()
    severity = BASE_SEVERITY * np.where(diesel, DIESEL_RELATIVITY, 1.0)
    df["ClaimNb"] = rng.poisson(0.5, size=len(df))
    # The mean of ClaimNb gamma draws is gamma with ClaimNb times the shape.
    shape = GAMMA_SHAPE * df["ClaimNb"].to_numpy()
    with_claims = shape > 0
    average = np.full(len(df), np.nan)
    average[with_claims] = rng.gamma(
        shape[with_claims], severity[with_claims] / shape[with_claims]
    )
    df["AvgClaimAmount"] = df["AvgAttritionalAmount"] = average
    total = np.nan_to_num(average) * df["ClaimNb"]
    df["ClaimAmount"] = df["AttritionalAmount"] = total
    return df


def test_glm_recovers_known_relativity(simulated_policies: pd.DataFrame) -> None:
    model = SeverityModel(features=GLM_FEATURES).fit(simulated_policies)

    relativities = model.relativities()["relativity"]

    assert relativities["VehGas[Diesel]"] == pytest.approx(DIESEL_RELATIVITY, rel=0.1)


@BOTH
def test_predicts_for_policies_without_claims(
    simulated_policies: pd.DataFrame, features: list[str]
) -> None:
    model = SeverityModel(features=features).fit(simulated_policies)
    no_claims = simulated_policies[simulated_policies["ClaimNb"] == 0]

    severity = model.predict_severity(no_claims)

    assert len(severity) == len(no_claims)
    assert (severity > 0).all()


@BOTH
def test_large_losses_scale_severity_by_loading(
    simulated_policies: pd.DataFrame, features: list[str]
) -> None:
    with_large_loss = simulated_policies.copy()
    # One policy has a large loss worth 10% of all capped claim cost on top.
    excess = 0.1 * simulated_policies["AttritionalAmount"].sum()
    policy = simulated_policies.index[simulated_policies["ClaimNb"] > 0][0]
    with_large_loss.loc[policy, "ClaimAmount"] += excess

    base = SeverityModel(features=features).fit(simulated_policies)
    loaded = SeverityModel(features=features).fit(with_large_loss)

    assert base.loading == pytest.approx(1.0)
    assert loaded.loading == pytest.approx(1.1)
    np.testing.assert_allclose(
        loaded.predict_severity(simulated_policies),
        1.1 * base.predict_severity(simulated_policies),
    )
    base_cost = base.relativities()["relativity"]["intercept"]
    loaded_cost = loaded.relativities()["relativity"]["intercept"]
    assert loaded_cost == pytest.approx(1.1 * base_cost)


def test_constant_severity_is_mean_cost_per_claim(
    simulated_policies: pd.DataFrame,
) -> None:
    model = SeverityModel().fit(simulated_policies)

    severity = model.predict_severity(simulated_policies)
    claims = simulated_policies["ClaimNb"].sum()

    np.testing.assert_allclose(
        severity, simulated_policies["ClaimAmount"].sum() / claims
    )
    relativities = model.relativities()
    assert relativities.index.tolist() == ["intercept"]
    assert relativities["relativity"]["intercept"] == pytest.approx(severity[0])
