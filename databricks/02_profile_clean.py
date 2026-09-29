# Databricks notebook source
# MAGIC %md
# MAGIC # Task 2 — Data Discovery, Profiling and Cleaning
# MAGIC **Project:** SkyPrint — Flight Tracking & Climate Impact Analysis
# MAGIC
# MAGIC Inputs (from Task 1, `data/raw/`):
# MAGIC - latest `opensky_<timestamp>.json` — OpenSky state vectors from the Saudi-area extraction
# MAGIC - latest `openmeteo_<timestamp>.json` — weather for aircraft-relevant grid locations and pressure levels
# MAGIC - `aircraftDatabase.csv` — aircraft lookup (`plane_id` / ICAO24 → aircraft metadata and `typecode`)
# MAGIC
# MAGIC This notebook keeps the same Task 2 methodology:
# MAGIC **Load → Flatten → Profile → Document issues → Clean → Save**
# MAGIC
# MAGIC Outputs:
# MAGIC - `data/interim/cleaned_opensky.csv`
# MAGIC - `data/interim/cleaned_weather.csv`
# MAGIC - `data/interim/cleaned_aircraft.csv`
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np
import json

pd.set_option("display.max_columns", None)

# Azure Data Lake paths
BRONZE_ROOT = "abfss://bronze@skyprint74815.dfs.core.windows.net"
SILVER_ROOT = "abfss://silver@skyprint74815.dfs.core.windows.net"


def profile(df):
    rows = []

    for c in df.columns:
        s = df[c]
        sample = s.dropna().iloc[0] if s.notna().any() else None

        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "percent_null": round(
                100 * s.isna().sum() / len(s), 2
            ) if len(s) else 0,
            "unique_count": int(s.nunique(dropna=True)),
            "sample_value": sample
        })

    return pd.DataFrame(rows)


def raw_json_files(prefix):
    """
    Find JSON files recursively inside the Bronze source folder.
    Example:
    bronze/opensky/year=2026/month=09/day=20/*.json
    """

    source_root = f"{BRONZE_ROOT}/{prefix}/"

    files = []

    def scan(path):
        for item in dbutils.fs.ls(path):
            if item.isDir():
                scan(item.path)
            elif item.path.lower().endswith(".json"):
                files.append(item.path)

    scan(source_root)

    files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"No {prefix} JSON files found in {source_root}"
        )

    return files


def batch_id_from_path(path, prefix):
    # ADLS path is a string, so extract filename manually
    filename = path.rstrip("/").split("/")[-1]
    stem = filename.rsplit(".", 1)[0]

    value = stem.removeprefix(f"{prefix}_")

    try:
        return pd.to_datetime(
            value,
            format="%Y-%m-%d_%H-%M-%S"
        ).strftime("%Y%m%d_%H%M%S")
    except (ValueError, TypeError):
        return value

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. OpenSky — Load and Flatten
# MAGIC
# MAGIC `/states/all` returns a JSON object with `time` and a nested `states` list.
# MAGIC Task 1 uses `extended=1`, so each state vector has 18 fields including `category`.
# MAGIC

# COMMAND ----------

opensky_files = raw_json_files("opensky")

print("OpenSky raw batches found:", len(opensky_files))
print("first:", opensky_files[0])
print("last :", opensky_files[-1])


# COMMAND ----------

OPENSKY_COLUMNS = [
    'plane_id', 'flight_id', 'origin_country', 'time_position', 'last_contact',
    'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity',
    'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk',
    'spi', 'source_type', 'category'
]

sky_frames = []

for opensky_path in opensky_files:

    # Read JSON directly from ADLS
    json_text = dbutils.fs.head(opensky_path, 1024 * 1024)
    opensky_raw = json.loads(json_text)

    states = opensky_raw.get('states', []) or []

    state_lengths = (
        pd.Series([len(row) for row in states])
        .value_counts()
        .sort_index()
    )

    if len(state_lengths) and not (
        len(state_lengths) == 1 and state_lengths.index[0] == 18
    ):
        raise ValueError(
            f'Unexpected OpenSky state-vector length in {opensky_path}. '
            'Expected 18 fields.'
        )

    frame = pd.DataFrame(states, columns=OPENSKY_COLUMNS)

    frame['batch_id'] = batch_id_from_path(
        opensky_path, 'opensky'
    )

    # Keep ADLS source path for traceability
    frame['source_file'] = opensky_path

    sky_frames.append(frame)

    # Extract filename from ADLS path
    filename = opensky_path.rstrip('/').split('/')[-1]

    print('loaded:', filename, '| rows:', len(frame))


df_sky = pd.concat(sky_frames, ignore_index=True)

print('combined shape:', df_sky.shape)

df_sky.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. OpenSky — Profiling
# MAGIC

# COMMAND ----------

profile_sky = profile(df_sky)
profile_sky


# COMMAND ----------

print('rows:', len(df_sky), ' columns:', df_sky.shape[1])
print('exact duplicate rows:', df_sky.duplicated().sum())
print('duplicate (batch_id, plane_id, time_position) rows:',
      df_sky.duplicated(subset=['batch_id', 'plane_id', 'time_position']).sum())

flight_text = df_sky['flight_id'].astype('string')
print('empty-string flight_ids:', (flight_text.str.strip() == '').sum())
print('missing positions:', df_sky[['latitude', 'longitude']].isna().any(axis=1).sum())
print('category value counts:')
print(df_sky['category'].value_counts(dropna=False).sort_index())


# COMMAND ----------

# MAGIC %md
# MAGIC ### Issues found — OpenSky
# MAGIC
# MAGIC | # | Issue | Decision |
# MAGIC |---|---|---|
# MAGIC | 1 | `sensors` is not useful for this project and is commonly null in this endpoint | Drop |
# MAGIC | 2 | `flight_id` may contain fixed-width trailing spaces or empty strings | Strip whitespace; empty → missing |
# MAGIC | 3 | Position/time/altitude fields can legitimately be missing in a live state vector | Keep missing values; do not invent/impute flight data |
# MAGIC | 4 | `squawk` and `spi` are operational fields not needed for SkyPrint fuel/trajectory analysis | Drop from cleaned analytical file; raw JSON remains unchanged |
# MAGIC | 5 | Unix timestamps are not analysis-friendly | Convert to UTC datetime |
# MAGIC | 6 | Future repeated pulls can create duplicate aircraft observations | Deduplicate on (`batch_id`, `plane_id`, `time_position`) |
# MAGIC | 7 | `plane_id` is the ICAO24 join key | Normalize to lowercase and trim whitespace |
# MAGIC | 8 | `category` is available because Task 1 uses `extended=1` | Keep |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. OpenSky — Cleaning
# MAGIC

# COMMAND ----------

df_sky_clean = df_sky.copy()

# 1) Remove columns outside the analytical scope.
df_sky_clean = df_sky_clean.drop(columns=['sensors', 'squawk', 'spi'])

# 2) Standardize identifiers/text.
df_sky_clean['plane_id'] = (
    df_sky_clean['plane_id'].astype('string').str.strip().str.lower()
)
df_sky_clean['flight_id'] = (
    df_sky_clean['flight_id'].astype('string').str.strip().replace('', pd.NA)
)
df_sky_clean['origin_country'] = (
    df_sky_clean['origin_country'].astype('string').str.strip().replace('', pd.NA)
)

# 3) Convert Unix epoch seconds to UTC datetimes.
for col in ['time_position', 'last_contact']:
    df_sky_clean[col] = pd.to_datetime(
        df_sky_clean[col], unit='s', utc=True, errors='coerce'
    )

# 4) Explicit dtypes.
df_sky_clean['on_ground'] = df_sky_clean['on_ground'].astype('boolean')
df_sky_clean['category'] = pd.to_numeric(df_sky_clean['category'], errors='coerce').astype('Int64')
df_sky_clean['source_type'] = pd.to_numeric(df_sky_clean['source_type'], errors='coerce').astype('Int64')

# 5) Deduplicate the observation natural key.
before = len(df_sky_clean)
df_sky_clean = df_sky_clean.drop_duplicates(subset=['batch_id', 'plane_id', 'time_position'])
print(f'dropped {before - len(df_sky_clean)} duplicate OpenSky observations')

df_sky_clean.dtypes


# COMMAND ----------

df_sky_clean.head()


# COMMAND ----------

sky_output = (
    "abfss://silver@skyprint74815.dfs.core.windows.net/"
    "opensky/cleaned_opensky.csv"
)

csv_text = df_sky_clean.to_csv(index=False)

dbutils.fs.put(
    sky_output,
    csv_text,
    overwrite=True
)

print("saved:", sky_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Open-Meteo — Load and Flatten
# MAGIC
# MAGIC The new extraction can request **multiple aircraft-relevant locations** in one call.
# MAGIC Open-Meteo therefore returns either:
# MAGIC - one forecast object (single location), or
# MAGIC - a list of forecast objects (multiple locations).
# MAGIC
# MAGIC Each forecast object contains hourly parallel arrays. We flatten every location and preserve its latitude/longitude.
# MAGIC

# COMMAND ----------

meteo_files = raw_json_files("openmeteo")

print("Open-Meteo raw batches found:", len(meteo_files))
print("first:", meteo_files[0])
print("last :", meteo_files[-1])


# COMMAND ----------

weather_frames = []

for meteo_path in meteo_files:

    # Read JSON directly from Bronze ADLS
    json_text = dbutils.fs.head(meteo_path, 1024 * 1024)
    meteo_raw = json.loads(json_text)

    meteo_locations = meteo_raw if isinstance(meteo_raw, list) else [meteo_raw]

    batch_id = batch_id_from_path(meteo_path, 'openmeteo')

    for location_index, location in enumerate(meteo_locations):
        hourly = location.get('hourly', {})

        if not hourly or 'time' not in hourly:
            continue

        frame = pd.DataFrame(hourly)

        frame['latitude'] = location.get('latitude')
        frame['longitude'] = location.get('longitude')
        frame['elevation_m'] = location.get('elevation')

        # Open-Meteo may not provide location_id;
        # fallback is unique within this batch.
        frame['location_id'] = location.get(
            'location_id',
            location_index
        )

        frame['timezone'] = location.get('timezone')
        frame['batch_id'] = batch_id

        # Keep full Bronze ADLS path for traceability
        frame['source_file'] = meteo_path

        weather_frames.append(frame)

    filename = meteo_path.rstrip('/').split('/')[-1]

    print(
        'loaded:',
        filename,
        '| locations:',
        len(meteo_locations)
    )


if not weather_frames:
    raise ValueError(
        'No hourly weather data found in the Open-Meteo raw responses.'
    )

df_wx = pd.concat(weather_frames, ignore_index=True)

print('combined shape:', df_wx.shape)
print('unique batches:', df_wx['batch_id'].nunique())
print(
    'unique weather coordinates:',
    df_wx[
        ['batch_id', 'latitude', 'longitude']
    ].drop_duplicates().shape[0]
)

df_wx.head()

# COMMAND ----------

print("Weather batches:", df_wx['batch_id'].nunique())
print("Weather location keys:", df_wx[['batch_id', 'location_id']].drop_duplicates().shape[0])
print("Unique returned coordinates across batches:", df_wx[['latitude', 'longitude']].drop_duplicates().shape[0])


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Open-Meteo — Profiling
# MAGIC

# COMMAND ----------

profile_wx = profile(df_wx)
profile_wx


# COMMAND ----------

print('rows:', len(df_wx))
print('exact duplicate rows:', df_wx.duplicated().sum())
print('unique locations:', df_wx[['latitude', 'longitude']].drop_duplicates().shape[0])
print('duplicate (batch, location, time) rows:',
      df_wx.duplicated(subset=['batch_id', 'latitude', 'longitude', 'time']).sum())
print('missing timestamps:', df_wx['time'].isna().sum())

# Show all rows involved in duplicated location + time keys
weather_dup_mask = df_wx.duplicated(
    subset=['batch_id', 'latitude', 'longitude', 'time'],
    keep=False
)

weather_duplicates = (
    df_wx[weather_dup_mask]
    .sort_values(['batch_id', 'latitude', 'longitude', 'time'])
)

print("Rows involved in duplicates:", len(weather_duplicates))
print(
    "Duplicated location/time groups:",
    weather_duplicates.groupby(
        ['batch_id', 'latitude', 'longitude', 'time']
    ).ngroups
)

weather_duplicates.head(20)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Issues found — Open-Meteo
# MAGIC
# MAGIC | # | Issue | Decision |
# MAGIC |---|---|---|
# MAGIC | 1 | New raw response contains multiple locations rather than the old single Riyadh point | Flatten every returned location |
# MAGIC | 2 | Hourly weather is stored as parallel arrays | Convert each location's `hourly` block to rows |
# MAGIC | 3 | `time` is text | Convert to UTC datetime |
# MAGIC | 4 | The same hour can appear at many locations, and repeated location/time keys may contain different weather values | Keep repeated (`latitude`, `longitude`, `time`) rows when weather values differ; remove only exact duplicate rows |
# MAGIC | 5 | Pressure-level variable names already encode the atmospheric level (`850hPa`, `700hPa`, etc.) | Keep them as separate columns for later aircraft-altitude matching |
# MAGIC | 6 | Missing weather values should not be guessed | Keep as missing; Task 3 validates them |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Open-Meteo — Cleaning
# MAGIC

# COMMAND ----------

df_wx_clean = df_wx.copy()

# 1) Convert time to UTC and give it an analysis-friendly name.
df_wx_clean['time'] = pd.to_datetime(df_wx_clean['time'], utc=True, errors='coerce')
df_wx_clean = df_wx_clean.rename(columns={'time': 'observation_time'})

# 2) Ensure coordinates/elevation and weather measurements are numeric.
non_numeric_weather = {'observation_time', 'timezone', 'batch_id', 'source_file'}
for col in df_wx_clean.columns:
    if col not in non_numeric_weather:
        df_wx_clean[col] = pd.to_numeric(df_wx_clean[col], errors='coerce')

# 3) Remove only true duplicates for the same location and hour.
# Do not drop rows only because returned latitude/longitude/time match.
# Some requested locations can map to the same Open-Meteo grid coordinates
# while retaining different location metadata.

exact_duplicates = df_wx_clean.duplicated().sum()

print("Exact duplicate weather rows:", exact_duplicates)

if exact_duplicates > 0:
    df_wx_clean = df_wx_clean.drop_duplicates()

print("Weather rows after cleaning:", len(df_wx_clean))


# COMMAND ----------

df_wx_clean.head()


# COMMAND ----------

weather_output = (
    "abfss://silver@skyprint74815.dfs.core.windows.net/"
    "openmeteo/cleaned_weather.csv"
)

csv_text = df_wx_clean.to_csv(index=False)

dbutils.fs.put(
    weather_output,
    csv_text,
    overwrite=True
)

print("saved:", weather_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Aircraft Type Lookup — Load and Profile
# MAGIC
# MAGIC The aircraft database is a reference dataset. Its `icao24` is renamed to `plane_id` so it can later join to OpenSky.
# MAGIC
# MAGIC **Important for SkyPrint:** `typecode` will later determine whether an aircraft type can be mapped to the fuel/performance model.  
# MAGIC We therefore clean `plane_id` and `typecode` carefully before resolving duplicate `plane_id` records.
# MAGIC

# COMMAND ----------

aircraft_path = (
    "abfss://bronze@skyprint74815.dfs.core.windows.net/"
    "aircraft/aircraftDatabase.csv"
)

df_ac = spark.read.option("header", "true").csv(aircraft_path).toPandas()

df_ac = df_ac.rename(columns={"icao24": "plane_id"})

print("shape:", df_ac.shape)
df_ac.head(3)

# COMMAND ----------

profile_ac = profile(df_ac)
profile_ac


# COMMAND ----------

null_pct = (df_ac.isna().sum() / len(df_ac) * 100).round(2).sort_values(ascending=False)

print('rows with plane_id null:', df_ac['plane_id'].isna().sum())
print('exact duplicate rows:', df_ac.duplicated().sum())
print('duplicate plane_id rows (before normalization):',
      df_ac[df_ac['plane_id'].notna()].duplicated(subset=['plane_id']).sum())
print('typecode nulls:', df_ac['typecode'].isna().sum())
print()
print('columns with >= 90% nulls:')
print(null_pct[null_pct >= 90])


# COMMAND ----------

# MAGIC %md
# MAGIC ### Issues found — Aircraft Database
# MAGIC
# MAGIC | # | Issue | Decision |
# MAGIC |---|---|---|
# MAGIC | 1 | Rows without `plane_id` cannot join to OpenSky | Drop |
# MAGIC | 2 | `plane_id` may differ only by case/whitespace | Trim + lowercase before duplicate checks |
# MAGIC | 3 | `typecode` may differ only by case/whitespace (for example `b738`, `B738 `) | Trim + uppercase before comparing |
# MAGIC | 4 | Blank-like text can look like a real value | Convert empty/`nan`/`none`/`null` text to missing |
# MAGIC | 5 | Duplicate `plane_id` rows may contain the same useful `typecode` plus missing values | Treat as one aircraft; keep the one normalized non-null typecode |
# MAGIC | 6 | Duplicate `plane_id` rows may contain genuinely different non-null `typecode`s | Do **not** guess; set `typecode` missing and `typecode_status = conflict` |
# MAGIC | 7 | A `plane_id` with no usable `typecode` | Keep aircraft row with `typecode_status = missing` |
# MAGIC | 8 | Some columns are zero-variance or ≥90% null and were already out of project scope | Keep the team's existing fixed drop list |
# MAGIC | 9 | `built`, `registered`, `reguntil` are date text | Convert to datetime |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Aircraft Type Lookup — Cleaning
# MAGIC

# COMMAND ----------

df_ac_clean = df_ac.copy()

# 1) Normalize missing plane IDs, then remove rows that cannot join to OpenSky.
df_ac_clean['plane_id'] = (
    df_ac_clean['plane_id']
    .astype('string')
    .str.strip()
    .str.lower()
    .replace({'': pd.NA, 'nan': pd.NA, 'none': pd.NA, 'null': pd.NA})
)

df_ac_clean = df_ac_clean.dropna(subset=['plane_id'])


# 2) Normalize typecode BEFORE deciding whether duplicate plane IDs conflict.
# We intentionally normalize case/whitespace only; we do not remove punctuation blindly.
df_ac_clean['typecode'] = (
    df_ac_clean['typecode']
    .astype('string')
    .str.strip()
    .str.upper()
    .replace({'': pd.NA, 'NAN': pd.NA, 'NONE': pd.NA, 'NULL': pd.NA})
)


# 3) Standardize manufacturer names.
manufacturer_map = {
    "Airbus": "Airbus",
    "Airbus Industrie": "Airbus",
    "Airbus Industries": "Airbus",
    "Airbus Military": "Airbus",
    "Airbus Sas": "Airbus",

    "Boeing": "Boeing",
    "The Boeing Co.": "Boeing",
    "Boeing Company": "Boeing",
    "The Boeing Company": "Boeing",
    "Boeing Co": "Boeing",
    "Boeing Commercial Airplane Group": "Boeing",
    "Boeing Commercial Aircraft": "Boeing",
    "Boeing Charleston (chs)": "Boeing",

    "Bombardier": "Bombardier",
    "Bombardier Inc": "Bombardier",
    "Emmbardier": "Bombardier",

    "Gulfstream Aerospace": "Gulfstream Aerospace",
    "Gulfstream Aerospace Corp": "Gulfstream Aerospace",
    "Gulfstream Aerospace Corporation": "Gulfstream Aerospace",

    "Cirrus": "Cirrus Aircraft",
    "Cirrus Aircraft": "Cirrus Aircraft",

    "Pilatus": "Pilatus",
    "Pilatus Aircraft Ltd": "Pilatus",

    "Beech": "Beechcraft",
    "Beechcraft": "Beechcraft",
    "Beechcraft Corp": "Beechcraft",

    "Leonardo": "Leonardo",
    "Leonardo Spa": "Leonardo",

    "De Havilland Canada": "De Havilland Canada",
    "De Havilland Aircraft Of Canada Inc.": "De Havilland Canada",

    "Cessna Aircraft Company": "Cessna",
    "Reims/cessna": "Cessna",

    "Piper Aircraft Corporation": "Piper",
    "Piper Aircraft Inc": "Piper",

    "Cirrus Design Corp": "Cirrus Aircraft",

    "Robinson Helicopter Company": "Robinson",
    "Robinson Helicopter Co": "Robinson",

    "Robinson Helicopter": "Robinson",

    "Beech Aircraft Corporation": "Beechcraft",

    "Dassault Aviation": "Dassault",

    "Diamond Aircraft Ind Inc": "Diamond Aircraft",

    "Avions De Transport Regional": "ATR"
}

df_ac_clean['manufacturername'] = (
    df_ac_clean['manufacturername']
    .astype('string')
    .str.strip()
    .replace({
        '': pd.NA,
        'nan': pd.NA,
        'none': pd.NA,
        'null': pd.NA
    })
    .replace(manufacturer_map)
)


# 4) Drop exact duplicates after normalization.
before = len(df_ac_clean)

df_ac_clean = df_ac_clean.drop_duplicates()

print(
    f'dropped {before - len(df_ac_clean)} '
    'exact duplicate aircraft rows after normalization'
)


# 5) Drop columns that are clearly unusable/out of scope for this project.
cols_to_drop = [
    'modes',
    'adsb',
    'acars',
    'status',
    'seatconfiguration',
    'firstflightdate',
    'testreg',
    'notes',
    'linenumber',
    'operatoriata',
    'operatorcallsign'
]

cols_to_drop = [
    c for c in cols_to_drop
    if c in df_ac_clean.columns
]

df_ac_clean = df_ac_clean.drop(columns=cols_to_drop)


# 6) Parse date-like text columns that survived.
for col in ['built', 'registered', 'reguntil']:
    if col in df_ac_clean.columns:
        df_ac_clean[col] = pd.to_datetime(
            df_ac_clean[col],
            errors='coerce'
        )

# COMMAND ----------

print(
    df_ac_clean["manufacturername"]
    .dropna()
    .value_counts()
    .head(50)
    .to_string()
)

# COMMAND ----------

# Inspect duplicate plane_ids AFTER normalization — optimized version

duplicate_mask = df_ac_clean['plane_id'].duplicated(keep=False)
df_ac_duplicates = df_ac_clean.loc[
    duplicate_mask,
    ['plane_id', 'typecode']
].copy()

duplicate_plane_ids = df_ac_duplicates['plane_id'].unique()

print(
    'duplicate plane_ids after normalization:',
    len(duplicate_plane_ids)
)

duplicate_typecode_summary = (
    df_ac_duplicates
    .dropna(subset=['typecode'])
    .drop_duplicates(subset=['plane_id', 'typecode'])
    .groupby('plane_id')['typecode']
    .agg(list)
    .reset_index(name='unique_non_null_typecodes')
)

duplicate_typecode_summary['typecode_count'] = (
    duplicate_typecode_summary['unique_non_null_typecodes'].str.len()
)

duplicate_typecode_summary.head(20)

# COMMAND ----------

# Resolve only duplicated plane_ids — much faster than looping over all aircraft

duplicate_mask = df_ac_clean.duplicated(subset=['plane_id'], keep=False)

df_unique = df_ac_clean[~duplicate_mask].copy()
df_duplicates = df_ac_clean[duplicate_mask].copy()

# Normal rows
df_unique['typecode_status'] = np.where(
    df_unique['typecode'].notna(),
    'matched',
    'missing'
)
df_unique['source_rows'] = 1
df_unique['typecode_candidates'] = df_unique['typecode']


# Resolve only the duplicated plane_ids
resolved_duplicates = []

for plane_id, group in df_duplicates.groupby('plane_id'):

    valid_typecodes = sorted(set(group['typecode'].dropna()))

    # Keep the row with the most available metadata
    result = group.loc[group.notna().sum(axis=1).idxmax()].copy()

    if len(valid_typecodes) == 0:
        result['typecode'] = pd.NA
        result['typecode_status'] = 'missing'

    elif len(valid_typecodes) == 1:
        result['typecode'] = valid_typecodes[0]
        result['typecode_status'] = 'matched'

    else:
        result['typecode'] = pd.NA
        result['typecode_status'] = 'conflict'

    result['source_rows'] = len(group)
    result['typecode_candidates'] = (
        '|'.join(valid_typecodes) if valid_typecodes else pd.NA
    )

    resolved_duplicates.append(result)


df_resolved_duplicates = pd.DataFrame(resolved_duplicates)

# Put unique + resolved duplicated aircraft back together
df_ac_clean = pd.concat(
    [df_unique, df_resolved_duplicates],
    ignore_index=True
)

print('final aircraft rows:', len(df_ac_clean))
print('unique plane_ids:', df_ac_clean['plane_id'].nunique())

print('\ntypecode status:')
print(df_ac_clean['typecode_status'].value_counts())

print('\nremaining duplicate plane_ids:',
      df_ac_clean.duplicated(subset=['plane_id']).sum())


# COMMAND ----------

# Show any real conflicts for review. They remain in the dataset, but typecode is not guessed.
conflicts = df_ac_clean[df_ac_clean['typecode_status'] == 'conflict'][
    ['plane_id', 'typecode', 'typecode_candidates', 'source_rows']
]
print('real typecode conflicts:', len(conflicts))
conflicts.head(20)


# COMMAND ----------

df_ac_clean.head()


# COMMAND ----------

# Final check for issues 8 and 9

print("Columns after cleaning:")
print(df_ac_clean.columns.tolist())

print("\nDate column types:")
for col in ['built', 'registered', 'reguntil']:
    print(col, ":", df_ac_clean[col].dtype)


# COMMAND ----------

aircraft_output = (
    "abfss://silver@skyprint74815.dfs.core.windows.net/"
    "aircraft/cleaned_aircraft"
)

spark_ac_clean = spark.createDataFrame(df_ac_clean)

(
    spark_ac_clean
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(aircraft_output)
)

print("saved:", aircraft_output)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC

# COMMAND ----------

summary = pd.DataFrame([
    {
        'dataset': 'OpenSky states',
        'raw_rows': len(df_sky),
        'cleaned_rows': len(df_sky_clean),
        'output': 'abfss://silver@skyprint74815.dfs.core.windows.net/opensky/cleaned_opensky.csv'
    },
    {
        'dataset': 'Open-Meteo hourly by location',
        'raw_rows': len(df_wx),
        'cleaned_rows': len(df_wx_clean),
        'output': 'abfss://silver@skyprint74815.dfs.core.windows.net/openmeteo/cleaned_weather.csv'
    },
    {
        'dataset': 'Aircraft type lookup',
        'raw_rows': len(df_ac),
        'cleaned_rows': len(df_ac_clean),
        'output': 'abfss://silver@skyprint74815.dfs.core.windows.net/aircraft/cleaned_aircraft'
    }
])

summary

# COMMAND ----------

print(df_ac_clean.columns.tolist())


# COMMAND ----------


print("=== FINAL CLEANING CHECK ===")
print("----------------------------")

# OpenSky
print("OpenSky rows:", len(df_sky_clean))
print(
    "OpenSky duplicate keys:",
    df_sky_clean.duplicated(
        ['batch_id', 'plane_id', 'time_position']
    ).sum()
)

# Weather
print("\nWeather rows:", len(df_wx_clean))
print(
    "Weather location IDs:",
    df_wx_clean['location_id'].nunique()
)
print(
    "Weather exact duplicate rows:",
    df_wx_clean.duplicated().sum()
)

# Aircraft
print("\nAircraft rows:", len(df_ac_clean))
print(
    "Aircraft duplicate IDs:",
    df_ac_clean['plane_id'].duplicated().sum()
)
print(
    "Aircraft typecode conflicts:",
    (df_ac_clean['typecode_status'] == 'conflict').sum()
)

print("\nTypecode status:")
print(
    df_ac_clean['typecode_status']
    .value_counts(dropna=False)
)