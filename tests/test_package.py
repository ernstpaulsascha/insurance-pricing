import insurance_pricing


def test_package_is_installed_with_version() -> None:
    assert insurance_pricing.__version__ == "0.1.0"
