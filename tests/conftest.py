import numpy as np
import pandas as pd
import pytest

from insurance_pricing.data.clean import preprocess
from insurance_pricing.features.rating_factors import add_rating_factors


@pytest.fixture
def raw_frequency() -> pd.DataFrame:
    """Policy table shaped like freMTPL2freq straight out of ``read_arff``."""
    return pd.DataFrame(
        {
            "IDpol": [1.0, 2.0, 3.0, 4.0],
            "ClaimNb": [1.0, 0.0, 2.0, 7.0],
            "Exposure": [0.5, 1.0, 2.01, 0.25],
            "Area": pd.Categorical(["A", "B", "A", "C"], categories=list("ABC")),
            "VehPower": [5.0, 6.0, 7.0, 4.0],
            "VehAge": [0.0, 3.0, 10.0, 1.0],
            "DrivAge": [55.0, 30.0, 42.0, 19.0],
            "BonusMalus": [50.0, 80.0, 50.0, 100.0],
            "VehBrand": pd.Categorical(["B1", "B2", "B1", "B2"]),
            "VehGas": ["Regular", "Diesel", "Diesel", "Regular"],
            "Density": [1217.0, 54.0, 76.0, 3003.0],
            "Region": pd.Categorical(["R11", "R22", "R11", "R22"]),
        }
    )


@pytest.fixture
def raw_severity() -> pd.DataFrame:
    """Claims table shaped like freMTPL2sev straight out of ``read_arff``."""
    return pd.DataFrame(
        {
            # Policy 3 reports 2 claims but has none here; policy 4 has 5;
            # policy 99 is not in the policy table.
            "IDpol": [1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 99.0],
            "ClaimAmount": [1000.0, 100.0, 200.0, 300.0, 400.0, 500_000.0, 50.0],
        }
    )


@pytest.fixture
def policies(raw_frequency: pd.DataFrame, raw_severity: pd.DataFrame) -> pd.DataFrame:
    """Cleaned policy table."""
    return preprocess(raw_frequency, raw_severity)


@pytest.fixture
def simulated_features() -> pd.DataFrame:
    """20k random policies with rating factors and exposure, but no claims yet."""
    n = 20_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Exposure": rng.uniform(0.1, 1.0, size=n),
            "Area": pd.Categorical(rng.choice(["A", "B"], size=n)),
            "VehPower": rng.integers(4, 16, size=n),
            "VehAge": rng.integers(0, 25, size=n),
            "DrivAge": rng.integers(18, 90, size=n),
            "BonusMalus": rng.integers(50, 150, size=n),
            "Density": rng.integers(1, 27_000, size=n).astype(float),
            "VehBrand": pd.Categorical(rng.choice(["B1", "B2"], size=n)),
            "VehGas": pd.Categorical(
                rng.choice(["Regular", "Diesel"], size=n),
                categories=["Regular", "Diesel"],
            ),
            "Region": pd.Categorical(rng.choice(["R11", "R22"], size=n)),
        }
    )
    return add_rating_factors(df)


DIESEL_FREQUENCY_RELATIVITY = 1.5
DIESEL_SEVERITY_RELATIVITY = 1.3


@pytest.fixture
def simulated_claims(simulated_features: pd.DataFrame) -> pd.DataFrame:
    """Policies where diesel raises both claim frequency and claim cost."""
    rng = np.random.default_rng(1)
    df = simulated_features.copy()
    diesel = (df["VehGas"] == "Diesel").to_numpy()
    frequency = 0.5 * np.where(diesel, DIESEL_FREQUENCY_RELATIVITY, 1.0)
    severity = 1000.0 * np.where(diesel, DIESEL_SEVERITY_RELATIVITY, 1.0)
    claim_nb = rng.poisson(frequency * df["Exposure"])
    with_claims = claim_nb > 0
    average = np.full(len(df), np.nan)
    shape = 2.0 * claim_nb[with_claims]
    average[with_claims] = rng.gamma(shape, severity[with_claims] / shape)

    df["ClaimNb"] = claim_nb
    df["Frequency"] = claim_nb / df["Exposure"]
    df["AvgClaimAmount"] = df["AvgAttritionalAmount"] = average
    total = np.nan_to_num(average) * claim_nb
    df["ClaimAmount"] = df["AttritionalAmount"] = total
    return df
