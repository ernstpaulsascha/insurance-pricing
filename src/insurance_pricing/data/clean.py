"""Clean the raw freMTPL2 tables and join them into one row per policy.

The cleaning follows the scikit-learn Tweedie example on the same dataset:
https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
"""

import pandas as pd

INTEGER_COLUMNS = ["IDpol", "ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus"]
CATEGORICAL_COLUMNS = ["Area", "VehBrand", "VehGas", "Region"]

# A handful of policies report exposures above one year, which is a data error.
MAX_EXPOSURE = 1.0
# Claim counts above 4 are very rare outliers (max 16) that destabilise a Poisson fit.
MAX_CLAIM_NB = 4
# Claims above this are large losses. A few of them (max ~4M) would dominate
# a gamma fit, so severity is modelled on amounts capped here and the excess
# is added back as a flat loading.
LARGE_LOSS_THRESHOLD = 50_000.0


def clean_frequency(freq: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and cap exposure in the policy table."""
    df = freq.copy()
    df[INTEGER_COLUMNS] = df[INTEGER_COLUMNS].astype("int64")
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].astype("category")
    df["Exposure"] = df["Exposure"].clip(upper=MAX_EXPOSURE)
    return df


def clean_severity(sev: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes in the claims table."""
    df = sev.copy()
    df["IDpol"] = df["IDpol"].astype("int64")
    return df


def build_policy_table(
    freq: pd.DataFrame,
    sev: pd.DataFrame,
    large_loss_threshold: float = LARGE_LOSS_THRESHOLD,
) -> pd.DataFrame:
    """Join cleaned policy and claim tables into one row per policy.

    ``ClaimNb`` in the raw policy table counts claims for which the severity
    table has no amount (about 9k policies), which would bias frequency against
    severity. ``ClaimNb`` is therefore recounted from the severity table, so
    both models see the same claims. Claims for policies missing from the policy
    table are dropped.

    Adds the modelling targets:

    - ``ClaimAmount``: total claim amount of the policy.
    - ``AttritionalAmount``: total claim amount with each claim capped at
      ``large_loss_threshold``.
    - ``Frequency``: claims per year of exposure (Poisson target, weight
      ``Exposure``).
    - ``AvgClaimAmount``: mean amount per claim, NaN without claims (gamma
      target, weight ``ClaimNb``).
    - ``AvgAttritionalAmount``: mean capped amount per claim, NaN without claims.
    - ``PurePremium``: claim amount per year of exposure (Tweedie target).
    """
    claims = (
        sev.assign(
            AttritionalAmount=sev["ClaimAmount"].clip(upper=large_loss_threshold)
        )
        .groupby("IDpol")
        .agg(
            ClaimNb=("ClaimAmount", "size"),
            ClaimAmount=("ClaimAmount", "sum"),
            AttritionalAmount=("AttritionalAmount", "sum"),
        )
    )
    df = freq.drop(columns="ClaimNb").join(claims, on="IDpol", validate="1:1")
    df["ClaimNb"] = df["ClaimNb"].fillna(0).astype("int64").clip(upper=MAX_CLAIM_NB)
    df[["ClaimAmount", "AttritionalAmount"]] = df[
        ["ClaimAmount", "AttritionalAmount"]
    ].fillna(0.0)

    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    claim_nb = df["ClaimNb"].where(df["ClaimNb"] > 0)
    df["AvgClaimAmount"] = df["ClaimAmount"] / claim_nb
    df["AvgAttritionalAmount"] = df["AttritionalAmount"] / claim_nb
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
    return df.reset_index(drop=True)


def preprocess(
    freq: pd.DataFrame,
    sev: pd.DataFrame,
    large_loss_threshold: float = LARGE_LOSS_THRESHOLD,
) -> pd.DataFrame:
    """Clean the raw tables and build the policy table."""
    return build_policy_table(
        clean_frequency(freq), clean_severity(sev), large_loss_threshold
    )
