from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_clean_outputs_and_keys():
    interim=ROOT/"data"/"interim"
    sky=pd.read_csv(interim/"cleaned_opensky.csv")
    wx=pd.read_csv(interim/"cleaned_weather.csv")
    ac=pd.read_csv(interim/"cleaned_aircraft.csv")
    assert sky.duplicated(["plane_id","time_position"]).sum()==0
    assert wx.duplicated(["location_id","observation_time"]).sum()==0
    assert ac["plane_id"].duplicated().sum()==0
