from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FREQUENCY_FILE = RAW_DIR / "freMTPL2freq.arff"
SEVERITY_FILE = RAW_DIR / "freMTPL2sev.arff"
