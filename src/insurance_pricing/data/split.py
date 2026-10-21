"""Split the policy table into train and test sets."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# freMTPL2 often splits one contract into several rows (consecutive IDpol,
# identical features). Rows sharing all of these belong to the same group.
GROUP_COLUMNS = [
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


def policy_groups(df: pd.DataFrame) -> pd.Series:
    """Label rows with identical rating features with the same group id."""
    return df.groupby(GROUP_COLUMNS, observed=True, sort=False).ngroup()


def train_test_split_policies(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split policies into train and test sets, keeping each group on one side.

    A plain random split would put copies of the same contract in both sets
    and overstate test performance. ``test_size`` is the share of groups, so
    the share of rows is only approximately ``test_size``.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, groups=policy_groups(df)))
    return df.iloc[train_idx], df.iloc[test_idx]
