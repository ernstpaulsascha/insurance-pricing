"""Turn raw policy features into GLM rating factors.

The driver age groups follow Noll, Salzmann & Wüthrich (2018), "Case Study:
French Motor Third-Party Liability Claims", on the same dataset. Vehicle age and
bonus-malus were refined after the GLM's actual/expected claims by raw value
showed misfit, and the changes were confirmed by 5-fold grouped cross-validation
on the training data. All bins are fixed in advance rather than fitted, so the
transform can be applied to train and test sets alike without leaking
information.
"""

import numpy as np
import pandas as pd

# Few vehicles above power 9; pool them into one level.
MAX_VEH_POWER = 9
VEH_AGE_BINS = [0, 1, 2, 3, 5, 10, 15, 20, np.inf]
VEH_AGE_LABELS = ["0", "1", "2", "3-4", "5-9", "10-14", "15-19", "20+"]
DRIV_AGE_BINS = [18, 21, 26, 31, 41, 51, 71, np.inf]
DRIV_AGE_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
# Fewer than 1 in 3,000 policies have a bonus-malus above 150.
MAX_BONUS_MALUS = 150
# The French bonus-malus coefficient (x100) falls by 5% per claim-free year, so
# a driver without claims only ever takes these values. Any other value comes
# from a recent claim penalty, and such drivers claim two to five times as often
# as their neighbours on the ladder.
BONUS_MALUS_LADDER = [50, 51, 54, 57, 60, 64, 68, 72, 76, 80, 85, 90, 95, 100]

# Area is not used: it is a banding of Density, which LogDensity already covers.
GLM_FEATURES = [
    "VehPowerGroup",
    "VehAgeGroup",
    "DrivAgeGroup",
    "LogBonusMalus",
    "BonusMalusOffLadder",
    "LogDensity",
    "VehBrand",
    "VehGas",
    "Region",
]
# Severity uses no rating factors: in 5-fold grouped cross-validation on capped
# claim amounts, no feature set beat a constant cost per claim by more than the
# fold-to-fold noise.
SEVERITY_FEATURES: list[str] = []
# Tree models handle raw values and do their own splitting.
GBM_FEATURES = [
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
]


def _bin(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(values, bins=bins, labels=labels, right=False, ordered=True)


def add_rating_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered GLM rating factors next to the raw features."""
    df = df.copy()
    power_levels = list(range(4, MAX_VEH_POWER + 1))
    df["VehPowerGroup"] = pd.Categorical(
        df["VehPower"].clip(upper=MAX_VEH_POWER), categories=power_levels
    )
    df["VehAgeGroup"] = _bin(df["VehAge"], VEH_AGE_BINS, VEH_AGE_LABELS)
    df["DrivAgeGroup"] = _bin(df["DrivAge"], DRIV_AGE_BINS, DRIV_AGE_LABELS)
    df["LogBonusMalus"] = np.log(df["BonusMalus"].clip(upper=MAX_BONUS_MALUS))
    df["BonusMalusOffLadder"] = (~df["BonusMalus"].isin(BONUS_MALUS_LADDER)).astype(
        float
    )
    df["LogDensity"] = np.log(df["Density"])
    return df
