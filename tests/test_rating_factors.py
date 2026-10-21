import numpy as np
import pandas as pd

from insurance_pricing.features.rating_factors import (
    GBM_FEATURES,
    GLM_FEATURES,
    add_rating_factors,
)


def test_add_rating_factors_adds_glm_features(policies: pd.DataFrame) -> None:
    df = add_rating_factors(policies)

    assert set(GLM_FEATURES) <= set(df.columns)
    assert set(GBM_FEATURES) <= set(df.columns)
    assert df[GLM_FEATURES].notna().all().all()


def test_add_rating_factors_leaves_input_unchanged(policies: pd.DataFrame) -> None:
    before = policies.copy()

    add_rating_factors(policies)

    pd.testing.assert_frame_equal(policies, before)


def test_age_bins_include_lower_edge() -> None:
    df = pd.DataFrame(
        {
            "VehPower": [4, 9, 15],
            "VehAge": [0, 1, 20],
            "DrivAge": [18, 21, 71],
            "BonusMalus": [50, 150, 230],
            "Density": [1.0, np.e, 27000.0],
        }
    )

    df = add_rating_factors(df)

    assert df["VehPowerGroup"].tolist() == [4, 9, 9]
    assert df["VehAgeGroup"].tolist() == ["0", "1", "20+"]
    assert df["DrivAgeGroup"].tolist() == ["18-20", "21-25", "71+"]
    np.testing.assert_allclose(df["LogBonusMalus"], np.log([50, 150, 150]))
    assert df["LogDensity"].iloc[:2].tolist() == [0.0, 1.0]


def test_rating_factor_levels_do_not_depend_on_data(policies: pd.DataFrame) -> None:
    # Levels must match across train and test so the design matrices line up.
    head = add_rating_factors(policies.head(1))
    tail = add_rating_factors(policies.tail(1))

    for column in ["VehPowerGroup", "VehAgeGroup", "DrivAgeGroup"]:
        assert head[column].cat.categories.equals(tail[column].cat.categories)


def test_bonus_malus_off_ladder_flags_claim_penalties() -> None:
    # 50-100 on the claim-free ladder; 62 and 100+ only follow a claim.
    df = pd.DataFrame(
        {
            "VehPower": [5] * 5,
            "VehAge": [3] * 5,
            "DrivAge": [40] * 5,
            "BonusMalus": [50, 60, 62, 100, 125],
            "Density": [100.0] * 5,
        }
    )

    df = add_rating_factors(df)

    assert df["BonusMalusOffLadder"].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]
