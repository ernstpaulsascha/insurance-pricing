"""Read the raw freMTPL2 ARFF files into DataFrames."""

from pathlib import Path

import arff
import pandas as pd

from insurance_pricing.config import FREQUENCY_FILE, SEVERITY_FILE


def read_arff(path: Path) -> pd.DataFrame:
    """Read an ARFF file, typing nominal attributes as categoricals.

    Nominal attributes keep the full set of levels declared in the header, so
    every subset of the data (e.g. train/test splits) shares the same
    categories.
    """
    with open(path, encoding="utf-8") as file:
        parsed = arff.load(file)

    columns = [name for name, _ in parsed["attributes"]]
    df = pd.DataFrame(parsed["data"], columns=columns)
    for name, kind in parsed["attributes"]:
        if isinstance(kind, list):
            df[name] = pd.Categorical(df[name], categories=kind)
    return df


def load_frequency(path: Path = FREQUENCY_FILE) -> pd.DataFrame:
    """Load the policy table: one row per policy with exposure and claim count."""
    return read_arff(path)


def load_severity(path: Path = SEVERITY_FILE) -> pd.DataFrame:
    """Load the claims table: one row per claim with its amount."""
    return read_arff(path)
