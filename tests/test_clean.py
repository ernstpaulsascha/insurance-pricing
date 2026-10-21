import pandas as pd

from insurance_pricing.data.clean import (
    CATEGORICAL_COLUMNS,
    INTEGER_COLUMNS,
    LARGE_LOSS_THRESHOLD,
    clean_frequency,
    clean_severity,
    preprocess,
)


def test_clean_frequency_fixes_dtypes_and_caps_exposure(
    raw_frequency: pd.DataFrame,
) -> None:
    df = clean_frequency(raw_frequency)

    for column in INTEGER_COLUMNS:
        assert df[column].dtype == "int64"
    for column in CATEGORICAL_COLUMNS:
        assert isinstance(df[column].dtype, pd.CategoricalDtype)
    assert df["Exposure"].tolist() == [0.5, 1.0, 1.0, 0.25]


def test_clean_severity_keeps_full_claim_amounts(raw_severity: pd.DataFrame) -> None:
    df = clean_severity(raw_severity)

    assert df["IDpol"].dtype == "int64"
    assert df["ClaimAmount"].max() == 500_000.0


def test_preprocess_recounts_claims_from_severity(
    raw_frequency: pd.DataFrame, raw_severity: pd.DataFrame
) -> None:
    df = preprocess(raw_frequency, raw_severity).set_index("IDpol")

    # Policy 99 has claims but no policy row, so it is dropped.
    assert df.index.tolist() == [1, 2, 3, 4]
    # Policy 3's two claims have no amounts; policy 4's five are capped at 4.
    assert df["ClaimNb"].tolist() == [1, 0, 0, 4]
    assert df["ClaimAmount"].tolist() == [1000.0, 0.0, 0.0, 501_000.0]


def test_preprocess_caps_each_claim_at_large_loss_threshold(
    raw_frequency: pd.DataFrame, raw_severity: pd.DataFrame
) -> None:
    df = preprocess(raw_frequency, raw_severity).set_index("IDpol")

    # Only policy 4's 500k claim is above the threshold.
    assert df["AttritionalAmount"].tolist() == [
        1000.0,
        0.0,
        0.0,
        1000.0 + LARGE_LOSS_THRESHOLD,
    ]
    assert df.loc[4, "AvgAttritionalAmount"] == (1000.0 + LARGE_LOSS_THRESHOLD) / 4
    assert pd.isna(df.loc[2, "AvgAttritionalAmount"])


def test_preprocess_adds_modelling_targets(
    raw_frequency: pd.DataFrame, raw_severity: pd.DataFrame
) -> None:
    df = preprocess(raw_frequency, raw_severity).set_index("IDpol")

    assert df.loc[1, "Frequency"] == 2.0
    assert df.loc[1, "PurePremium"] == 2000.0
    assert df.loc[1, "AvgClaimAmount"] == 1000.0
    assert pd.isna(df.loc[2, "AvgClaimAmount"])
    assert df.loc[2, "Frequency"] == 0.0
