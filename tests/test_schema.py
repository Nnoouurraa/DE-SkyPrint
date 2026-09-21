from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_validated_outputs_exist_and_are_valid():
    interim=ROOT/"data"/"interim"
    for name in ["validated_opensky.csv","validated_weather.csv","validated_aircraft.csv"]:
        path=interim/name
        assert path.exists(), f"Missing {name}"
        df=pd.read_csv(path)
        assert len(df)>0
        if "schema_valid" in df.columns:
            assert df["schema_valid"].fillna(False).all()

def test_saudi_flag_exists():
    sky=pd.read_csv(ROOT/"data"/"interim"/"validated_opensky.csv")
    assert "inside_saudi_boundary" in sky.columns
    assert sky["inside_saudi_boundary"].notna().all()
