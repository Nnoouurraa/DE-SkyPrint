from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def test_raw_inputs_exist_after_extraction():
    raw=ROOT / "data" / "raw"
    assert list(raw.glob("opensky_*.json")), "No timestamped OpenSky JSON found"
    assert list(raw.glob("openmeteo_*.json")), "No timestamped Open-Meteo JSON found"
    assert (raw / "aircraftDatabase.csv").exists()

def test_latest_opensky_has_states():
    files=sorted((ROOT/"data"/"raw").glob("opensky_*.json"), key=lambda p:p.stat().st_mtime)
    if files:
        payload=json.loads(files[-1].read_text())
        assert "states" in payload
