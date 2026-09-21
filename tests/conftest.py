from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def final_df():
    path = ROOT / "data" / "processed" / "final.csv"
    if not path.exists():
        pytest.skip("Run `python main.py` first to generate final.csv")
    return pd.read_csv(path)
