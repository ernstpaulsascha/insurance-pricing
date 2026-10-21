import pandas as pd

from insurance_pricing.data.split import policy_groups, train_test_split_policies


def _with_duplicates(policies: pd.DataFrame, copies: int) -> pd.DataFrame:
    """Repeat each policy under new IDpols, as freMTPL2 does for split contracts."""
    df = pd.concat([policies] * copies, ignore_index=True)
    df["IDpol"] = range(len(df))
    return df


def test_policy_groups_match_identical_rating_features(
    policies: pd.DataFrame,
) -> None:
    df = _with_duplicates(policies, copies=3)

    groups = policy_groups(df)

    assert groups.nunique() == len(policies)
    assert (groups.groupby(df["DrivAge"]).nunique() == 1).all()


def test_split_keeps_each_group_on_one_side(policies: pd.DataFrame) -> None:
    df = _with_duplicates(policies, copies=5)
    groups = policy_groups(df)

    train, test = train_test_split_policies(df, test_size=0.25)

    assert len(train) + len(test) == len(df)
    assert set(train["IDpol"]).isdisjoint(test["IDpol"])
    assert set(groups[train.index]).isdisjoint(groups[test.index])
    # One of the four groups, with its five copies, goes to test.
    assert len(test) == 5


def test_split_is_reproducible(policies: pd.DataFrame) -> None:
    df = _with_duplicates(policies, copies=5)

    _, first = train_test_split_policies(df, random_state=0)
    _, second = train_test_split_policies(df, random_state=0)

    pd.testing.assert_frame_equal(first, second)
