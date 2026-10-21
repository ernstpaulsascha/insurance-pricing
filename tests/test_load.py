from pathlib import Path

import pandas as pd

from insurance_pricing.data.load import read_arff

ARFF = """\
@relation test
@attribute IDpol numeric
@attribute Area {'A','B','C'}
@attribute VehGas string
@data
1,'A','Regular'
2,'C','Diesel'
"""


def test_read_arff_types_nominal_attributes_as_categoricals(tmp_path: Path) -> None:
    path = tmp_path / "test.arff"
    path.write_text(ARFF, encoding="utf-8")

    df = read_arff(path)

    assert list(df.columns) == ["IDpol", "Area", "VehGas"]
    assert df["IDpol"].tolist() == [1.0, 2.0]
    assert isinstance(df["Area"].dtype, pd.CategoricalDtype)
    # All declared levels are kept, including ones absent from the data.
    assert df["Area"].cat.categories.tolist() == ["A", "B", "C"]
    assert df["VehGas"].tolist() == ["Regular", "Diesel"]
