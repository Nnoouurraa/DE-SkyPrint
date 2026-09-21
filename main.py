from pathlib import Path
from src.extract import run as run_extract
from src.clean import run as run_clean
from src.schema import run as run_schema
from src.transform import run as run_transform

ROOT = Path(__file__).resolve().parent

def _preflight():
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    aircraft_raw = ROOT / "data" / "raw" / "aircraftDatabase.csv"
    aircraft_ref = ROOT / "data" / "reference" / "aircraftDatabase.csv"
    if not aircraft_raw.exists() and aircraft_ref.exists():
        aircraft_raw.write_bytes(aircraft_ref.read_bytes())

    boundary = ROOT / "data" / "reference" / "geoBoundaries-SAU-ADM0.geojson"
    missing=[]
    if not aircraft_raw.exists():
        missing.append(str(aircraft_raw.relative_to(ROOT)))
    if not boundary.exists():
        missing.append(str(boundary.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError(
            "Required static/reference file(s) are missing: " + ", ".join(missing) +
            ". Keep these source/reference assets before running the pipeline."
        )

def main():
    _preflight()
    stages=[
        ("1/4 Extract", run_extract),
        ("2/4 Profile & Clean", run_clean),
        ("3/4 Schema & Validate", run_schema),
        ("4/4 Join & Transform", run_transform),
    ]
    for label, fn in stages:
        print(f"\n{'='*72}\n{label}\n{'='*72}")
        fn()
    final_path=ROOT / "data" / "processed" / "final.csv"
    if not final_path.exists():
        raise RuntimeError("Pipeline completed without producing data/processed/final.csv")
    print(f"\nPipeline complete: {final_path}")

if __name__ == "__main__":
    main()
