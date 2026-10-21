import pandas as pd

from insurance_pricing.data.clean import preprocess
from insurance_pricing.data.load import load_frequency, load_severity
from insurance_pricing.data.split import train_test_split_policies


def load_policy_data() -> pd.DataFrame:
    """Load the raw freMTPL2 files and return the cleaned policy table."""
    return preprocess(load_frequency(), load_severity())


__all__ = [
    "load_frequency",
    "load_policy_data",
    "load_severity",
    "preprocess",
    "train_test_split_policies",
]
