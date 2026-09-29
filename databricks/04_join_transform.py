# Databricks notebook source
# MAGIC %md
# MAGIC # Task 4 — Join, Transformation Rules, and Testing
# MAGIC
# MAGIC ## SkyPrint
# MAGIC
# MAGIC ### Objective
# MAGIC
# MAGIC Build a transformation pipeline that integrates validated flight, aircraft, and weather data for flights over Saudi Arabia.
# MAGIC
# MAGIC The final dataset will support two main analytical goals:
# MAGIC
# MAGIC 1. **Air Traffic Analysis**
# MAGIC    - Preserve flight position and time information for traffic-density analysis.
# MAGIC    - Support trajectory and route-overload analysis using repeated OpenSky observations across batches.
# MAGIC
# MAGIC 2. **Environmental Analysis**
# MAGIC    - Enrich flight observations with aircraft metadata and weather conditions.
# MAGIC    - Prepare valid observations for fuel-consumption and emissions estimation using OpenAP.
# MAGIC
# MAGIC ### Design Principles
# MAGIC
# MAGIC - Use validated outputs from Task 3.
# MAGIC - Keep the transformation compatible with repeated OpenSky snapshots.
# MAGIC - Document and test every join and transformation rule.
# MAGIC - Do not silently drop unmatched records.
# MAGIC - Check row counts and duplicate keys after joins.
# MAGIC - Apply exact Saudi Arabia polygon filtering in this stage.
# MAGIC - Match weather by spatial, temporal, and altitude context.
# MAGIC - Produce final analysis-ready outputs in the Gold layer.
# MAGIC
# MAGIC > Note: The OpenSky dataset contains repeated observations across multiple extraction batches. Task 4 therefore preserves batch and timestamp lineage so trajectory and traffic analysis can be performed without collapsing distinct observations.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Join Specification
# MAGIC
# MAGIC Task 4 uses the validated outputs produced by Task 3.
# MAGIC
# MAGIC ### Input Datasets
# MAGIC
# MAGIC | Dataset | Purpose | Key / Matching Method |
# MAGIC |---|---|---|
# MAGIC | Validated OpenSky | Base flight observations | `plane_id` |
# MAGIC | Validated Aircraft | Aircraft metadata and OpenAP type information | `plane_id` |
# MAGIC | Validated Weather | Atmospheric conditions | Spatial + temporal + altitude matching |
# MAGIC
# MAGIC ### Join 1 — OpenSky + Aircraft
# MAGIC
# MAGIC - **Base dataset:** OpenSky
# MAGIC - **Join key:** `plane_id`
# MAGIC - **Join type:** Left join
# MAGIC - **Relationship:** Many-to-one
# MAGIC - **Reason:** Preserve every valid flight observation even when aircraft metadata is unavailable.
# MAGIC - **Duplicate handling:** Aircraft `plane_id` must be unique before the join. Duplicate keys are treated as an error rather than silently removed.
# MAGIC
# MAGIC ### Join 2 — Flight + Weather
# MAGIC
# MAGIC Weather cannot be joined using `plane_id`.
# MAGIC
# MAGIC Each flight observation will be matched to weather using:
# MAGIC
# MAGIC - geographic location,
# MAGIC - observation time,
# MAGIC - and the atmospheric level appropriate to the aircraft altitude.
# MAGIC
# MAGIC **Matching policy:** select the best available weather observation using defined spatial, temporal, and altitude rules, while preserving unmatched flight observations with readiness flags.
# MAGIC
# MAGIC The matching logic will be implemented and tested explicitly before weather attributes are added.
# MAGIC
# MAGIC ### Row-count Policy
# MAGIC
# MAGIC - Row counts will be measured before and after every join.
# MAGIC - Enrichment joins must not unexpectedly multiply flight observations.
# MAGIC - Unmatched records will be retained and flagged rather than silently dropped.

# COMMAND ----------

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 100)

# ============================================================
# SkyPrint — Databricks / ADLS paths
# ============================================================

SILVER_BASE = (
    "abfss://silver@skyprint74815.dfs.core.windows.net"
)

GOLD_BASE = (
    "abfss://gold@skyprint74815.dfs.core.windows.net"
)

REFERENCE_BASE = (
    "abfss://reference@skyprint74815.dfs.core.windows.net"
)

# ------------------------------------------------------------
# Validated inputs produced by Task 3
# ------------------------------------------------------------

VALIDATION_BASE = (
    f"{SILVER_BASE}/validation"
)

OPENSKY_VALIDATED = (
    f"{VALIDATION_BASE}/opensky/validated"
)

WEATHER_VALIDATED = (
    f"{VALIDATION_BASE}/weather/validated"
)

AIRCRAFT_VALIDATED = (
    f"{VALIDATION_BASE}/aircraft/validated"
)

SKYPRINT_READINESS = (
    f"{VALIDATION_BASE}/skyprint_readiness"
)

# ------------------------------------------------------------
# Reference data
# ------------------------------------------------------------

SAUDI_BOUNDARY_PATH = (
    f"{REFERENCE_BASE}/geo/geoBoundaries-SAU-ADM0.geojson"
)

# ------------------------------------------------------------
# Task 4 Gold output root
# ------------------------------------------------------------

TRANSFORM_OUTPUT_BASE = (
    f"{GOLD_BASE}/skyprint"
)

print("Validated OpenSky :", OPENSKY_VALIDATED)
print("Validated Weather :", WEATHER_VALIDATED)
print("Validated Aircraft:", AIRCRAFT_VALIDATED)
print("Readiness input   :", SKYPRINT_READINESS)
print("Gold output base  :", TRANSFORM_OUTPUT_BASE)

# COMMAND ----------

# Check Task 3 validated outputs available for Task 4

task3_paths = {
    "OpenSky validated": OPENSKY_VALIDATED,
    "Weather validated": WEATHER_VALIDATED,
    "Aircraft validated": AIRCRAFT_VALIDATED,
    "SkyPrint readiness": SKYPRINT_READINESS,
}

print("Task 3 validated inputs:")

for name, path in task3_paths.items():
    try:
        dbutils.fs.ls(path)
        print(f"- {name}: available")
    except Exception as e:
        print(f"- {name}: missing or inaccessible")
        raise

# COMMAND ----------

sky = spark.read.option("header", "true").csv(OPENSKY_VALIDATED).toPandas()
aircraft = spark.read.option("header", "true").csv(AIRCRAFT_VALIDATED).toPandas()
weather = spark.read.option("header", "true").csv(WEATHER_VALIDATED).toPandas()

print("Validated OpenSky:", sky.shape)
print("Validated Aircraft:", aircraft.shape)
print("Validated Weather:", weather.shape)

# COMMAND ----------

# Inspect columns and join keys before transformation

print("=== OPENSKY COLUMNS ===")
print(sky.columns.tolist())

print("\n=== AIRCRAFT COLUMNS ===")
print(aircraft.columns.tolist())

print("\n=== WEATHER COLUMNS ===")
print(weather.columns.tolist())

print("\n=== KEY CHECKS ===")
print("OpenSky rows:", len(sky))
print("OpenSky plane_id nulls:", sky["plane_id"].isna().sum())

print("Aircraft rows:", len(aircraft))
print("Aircraft plane_id nulls:", aircraft["plane_id"].isna().sum())
print("Aircraft plane_id duplicates:", aircraft["plane_id"].duplicated().sum())

print("Weather rows:", len(weather))
print("Weather location_id nulls:", weather["location_id"].isna().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Join Flight and Aircraft Data
# MAGIC
# MAGIC Join validated OpenSky flight observations with validated aircraft metadata using `plane_id`.
# MAGIC
# MAGIC A left join is used to preserve all flight observations, including aircraft without matching metadata. The aircraft key must remain unique to prevent row multiplication.

# COMMAND ----------

# Join OpenSky flight observations with aircraft metadata

aircraft_cols = [
    "plane_id",
    "registration",
    "manufacturername",
    "model",
    "typecode",
    "engines",
    "categoryDescription",
    "openap_typecode",
    "openap_resolution",
    "openap_supported",
    "metadata_ready",
    "fuel_model_ready"
]

flight_aircraft = sky.merge(
    aircraft[aircraft_cols],
    on="plane_id",
    how="left",
    validate="many_to_one"
)

# Flag successful aircraft metadata matches
flight_aircraft["aircraft_metadata_matched"] = (
    flight_aircraft["metadata_ready"].fillna(False).astype(bool)
)

print("Rows before join:", len(sky))
print("Rows after join:", len(flight_aircraft))
print(
    "Aircraft metadata matched:",
    int(flight_aircraft["aircraft_metadata_matched"].sum())
)
print(
    "OpenAP ready:",
    int(
        flight_aircraft["openap_supported"]
        .astype("string")
        .str.lower()
        .eq("true")
        .sum()
    )
)

# COMMAND ----------

# Test Join 1

assert len(flight_aircraft) == len(sky), \
    "Join 1 failed: row count changed."

assert flight_aircraft["plane_id"].isna().sum() == 0, \
    "Join 1 failed: null plane_id found."

assert aircraft["plane_id"].duplicated().sum() == 0, \
    "Join 1 failed: duplicate aircraft keys found."

print("JOIN 1 TEST: PASS")
print("All OpenSky observations were preserved without row multiplication.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Match Flight and Weather Data
# MAGIC
# MAGIC Match each flight observation with the most appropriate weather observation using location and time.
# MAGIC
# MAGIC Because weather conditions are provided at multiple atmospheric pressure levels, the aircraft altitude will then be used to select the weather level that best represents the aircraft's flight conditions.
# MAGIC
# MAGIC The matching process must preserve all flight observations and avoid row multiplication.

# COMMAND ----------

# Inspect flight and weather matching fields

print("=== FLIGHT SAMPLE ===")
display(
    flight_aircraft[
        ["plane_id", "time_position", "latitude", "longitude",
         "baro_altitude", "geo_altitude"]
    ].head()
)

print("\n=== WEATHER SAMPLE ===")
display(
    weather[
        ["location_id", "observation_time", "latitude", "longitude"]
    ].head()
)

print("\nFlight time range:")
print(flight_aircraft["time_position"].min(), "->",
      flight_aircraft["time_position"].max())

print("\nWeather time range:")
print(weather["observation_time"].min(), "->",
      weather["observation_time"].max())

print("\nUnique weather locations:",
      weather["location_id"].nunique())

# COMMAND ----------

# Prepare timestamps for weather matching

flight_aircraft["time_position"] = pd.to_datetime(
    flight_aircraft["time_position"],
    utc=True
)

weather["observation_time"] = pd.to_datetime(
    weather["observation_time"],
    utc=True
)

# Match each flight observation to the nearest weather hour
flight_aircraft["weather_time"] = (
    flight_aircraft["time_position"].dt.round("h")
)

print(
    flight_aircraft[
        ["plane_id", "time_position", "weather_time"]
    ].head()
)

print(
    "\nWeather times available:",
    weather["observation_time"].nunique()
)

# COMMAND ----------

# Find the nearest weather location for each flight observation.
# IMPORTANT: candidates are restricted to the same extraction batch.

weather_locations = (
    weather[["batch_id", "location_id", "latitude", "longitude"]]
    .drop_duplicates(["batch_id", "location_id"])
    .reset_index(drop=True)
)

# Convert coordinates loaded from CSV into numeric values
for col in ["latitude", "longitude"]:
    flight_aircraft[col] = pd.to_numeric(
        flight_aircraft[col],
        errors="coerce"
    )

    weather_locations[col] = pd.to_numeric(
        weather_locations[col],
        errors="coerce"
    )


def find_nearest_weather_location(row):
    candidates = weather_locations[
        weather_locations["batch_id"] == row["batch_id"]
    ]

    if (
        candidates.empty
        or pd.isna(row["latitude"])
        or pd.isna(row["longitude"])
    ):
        return pd.NA

    distances = (
        (candidates["latitude"] - row["latitude"]) ** 2
        + (candidates["longitude"] - row["longitude"]) ** 2
    )

    nearest_index = distances.idxmin()

    return candidates.loc[
        nearest_index,
        "location_id"
    ]


flight_aircraft["weather_location_id"] = flight_aircraft.apply(
    find_nearest_weather_location,
    axis=1
)

print(
    flight_aircraft[
        [
            "batch_id",
            "plane_id",
            "latitude",
            "longitude",
            "weather_location_id"
        ]
    ].head()
)

print(
    "Flights with weather location:",
    flight_aircraft["weather_location_id"].notna().sum(),
    "/",
    len(flight_aircraft)
)

# COMMAND ----------

unmatched = flight_aircraft[
    flight_aircraft["weather_location_id"].isna()
]

weather_batches = set(weather["batch_id"])

print("Unmatched rows:", len(unmatched))
print("Unmatched batches:", unmatched["batch_id"].nunique())

print(
    "Unmatched rows whose batch exists in Weather:",
    unmatched["batch_id"].isin(weather_batches).sum()
)

print(
    "Flight batches missing completely from Weather:",
    sorted(set(unmatched["batch_id"]) - weather_batches)
)

# COMMAND ----------

# Calculate distance to the matched weather location

weather_location_lookup = weather_locations.rename(
    columns={
        "latitude": "weather_latitude",
        "longitude": "weather_longitude"
    }
)

flight_aircraft = flight_aircraft.merge(
    weather_location_lookup,
    left_on=["batch_id", "weather_location_id"],
    right_on=["batch_id", "location_id"],
    how="left",
    validate="many_to_one"
)

# Haversine distance in kilometers
lat1 = np.radians(flight_aircraft["latitude"])
lon1 = np.radians(flight_aircraft["longitude"])
lat2 = np.radians(flight_aircraft["weather_latitude"])
lon2 = np.radians(flight_aircraft["weather_longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
)

flight_aircraft["weather_distance_km"] = (
    6371 * 2 * np.arcsin(np.sqrt(a))
)

print(
    flight_aircraft[
        [
            "plane_id",
            "latitude",
            "longitude",
            "weather_latitude",
            "weather_longitude",
            "weather_distance_km"
        ]
    ].head()
)

print("\nMaximum distance (km):",
      round(flight_aircraft["weather_distance_km"].max(), 2))

print("Average distance (km):",
      round(flight_aircraft["weather_distance_km"].mean(), 2))


# COMMAND ----------

# Flag whether each flight has a matching Weather batch

weather_batches = set(
    weather["batch_id"].dropna().astype(str)
)

flight_aircraft["weather_batch_available"] = (
    flight_aircraft["batch_id"]
    .astype(str)
    .isin(weather_batches)
)

print("Flight rows preserved:", len(flight_aircraft))
print(
    "Weather batch available:",
    int(flight_aircraft["weather_batch_available"].sum())
)
print(
    "Weather batch missing:",
    int((~flight_aircraft["weather_batch_available"]).sum())
)

# COMMAND ----------

# Join flight observations with weather by location and time

flight_weather = flight_aircraft.merge(
    weather,
    left_on=["batch_id", "weather_location_id", "weather_time"],
    right_on=["batch_id", "location_id", "observation_time"],
    how="left",
    validate="many_to_one",
    suffixes=("", "_weather")
)

print("Rows before weather join:", len(flight_aircraft))
print("Rows after weather join:", len(flight_weather))

print(
    "Weather matched:",
    int(
        flight_weather["weather_ready"]
        .astype("string")
        .str.lower()
        .eq("true")
        .sum()
    ),
    "/",
    len(flight_weather)
)

# COMMAND ----------

# Test weather join

assert len(flight_weather) == len(flight_aircraft), \
    "Weather join failed: row count changed."

matched = flight_weather["observation_time"].notna()

weather_ready_bool = (
    flight_weather["weather_ready"]
    .astype("string")
    .str.lower()
    .eq("true")
)

assert weather_ready_bool[matched].all(), \
    "Weather join failed: some matched weather records are not ready."

assert (
    flight_weather.loc[matched, "weather_location_id"].astype(str)
    == flight_weather.loc[matched, "location_id"].astype(str)
).all(), "Weather join failed: location mismatch."

assert (
    flight_weather.loc[matched, "weather_time"]
    == flight_weather.loc[matched, "observation_time"]
).all(), "Weather join failed: time mismatch."

print("WEATHER JOIN TEST: PASS")
print("Flight rows preserved:", len(flight_weather))
print("Weather matched:", int(matched.sum()))
print("Weather unmatched:", int((~matched).sum()))

# COMMAND ----------

# Select the atmospheric pressure level nearest to each airborne flight altitude

pressure_levels = [850, 700, 500, 300, 250, 200]

# Convert altitude fields to numeric
for col in ["geo_altitude", "baro_altitude"]:
    flight_weather[col] = pd.to_numeric(
        flight_weather[col],
        errors="coerce"
    )

for level in pressure_levels:
    col = f"geopotential_height_{level}hPa"
    flight_weather[col] = pd.to_numeric(
        flight_weather[col],
        errors="coerce"
    )

# Prefer geometric altitude, fall back to barometric altitude
flight_weather["flight_altitude_m"] = (
    flight_weather["geo_altitude"]
    .fillna(flight_weather["baro_altitude"])
)

height_difference = pd.DataFrame(
    {
        level: abs(
            flight_weather[f"geopotential_height_{level}hPa"].to_numpy()
            - flight_weather["flight_altitude_m"].to_numpy()
        )
        for level in pressure_levels
    },
    index=flight_weather.index
)

valid_altitude = flight_weather["flight_altitude_m"].notna()

flight_weather["pressure_level_hpa"] = pd.NA

flight_weather.loc[valid_altitude, "pressure_level_hpa"] = (
    height_difference.loc[valid_altitude].idxmin(axis=1)
)

flight_weather["pressure_level_hpa"] = (
    flight_weather["pressure_level_hpa"].astype("Int64")
)

print("Total flight observations:", len(flight_weather))
print("Altitude available:", int(valid_altitude.sum()))
print("Altitude unavailable:", int((~valid_altitude).sum()))

print("\nSelected pressure levels:")
print(
    flight_weather["pressure_level_hpa"]
    .value_counts(dropna=False)
    .sort_index()
)

# COMMAND ----------

# Verify pressure-level selection against aircraft altitude

pressure_check = (
    flight_weather
    .dropna(subset=["pressure_level_hpa"])
    .groupby("pressure_level_hpa")["flight_altitude_m"]
    .agg(["count", "min", "mean", "max"])
    .round(1)
)

print("=== PRESSURE LEVEL vs FLIGHT ALTITUDE ===")
display(pressure_check)

assert pressure_check.index.isin([850, 700, 500, 300, 250, 200]).all()
assert pressure_check["count"].sum() == flight_weather["pressure_level_hpa"].notna().sum()

print("\nPRESSURE LEVEL CHECK: PASS")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Select Altitude-Matched Weather Conditions
# MAGIC
# MAGIC For each flight observation with a valid altitude, select temperature, wind speed, and wind direction from the atmospheric pressure level matched to the aircraft altitude.
# MAGIC
# MAGIC Flights without a valid altitude are retained, but altitude-dependent weather attributes remain unavailable.
# MAGIC
# MAGIC These derived weather attributes will later support wind-aware flight features and fuel/emissions estimation.

# COMMAND ----------

# Select weather variables from the pressure level matched to each flight altitude

pressure_levels = [850, 700, 500, 300, 250, 200]

for level in pressure_levels:
    for col in [
        f"temperature_{level}hPa",
        f"wind_speed_{level}hPa",
        f"wind_direction_{level}hPa",
    ]:
        flight_weather[col] = pd.to_numeric(
            flight_weather[col],
            errors="coerce"
        )

flight_weather["flight_temperature_c"] = np.nan
flight_weather["flight_wind_speed_kmh"] = np.nan
flight_weather["flight_wind_direction_deg"] = np.nan

for level in pressure_levels:
    mask = flight_weather["pressure_level_hpa"].eq(level)

    flight_weather.loc[mask, "flight_temperature_c"] = (
        flight_weather.loc[mask, f"temperature_{level}hPa"]
    )

    flight_weather.loc[mask, "flight_wind_speed_kmh"] = (
        flight_weather.loc[mask, f"wind_speed_{level}hPa"]
    )

    flight_weather.loc[mask, "flight_wind_direction_deg"] = (
        flight_weather.loc[mask, f"wind_direction_{level}hPa"]
    )

print("=== ALTITUDE-MATCHED WEATHER ===")

display(
    flight_weather[
        [
            "plane_id",
            "flight_altitude_m",
            "pressure_level_hpa",
            "flight_temperature_c",
            "flight_wind_speed_kmh",
            "flight_wind_direction_deg"
        ]
    ].head(10)
)

print("\nWeather attributes available:")
print("Temperature:", flight_weather["flight_temperature_c"].notna().sum())
print("Wind speed:", flight_weather["flight_wind_speed_kmh"].notna().sum())
print("Wind direction:", flight_weather["flight_wind_direction_deg"].notna().sum())

# COMMAND ----------

# Test altitude-matched weather attributes

valid_weather = flight_weather["pressure_level_hpa"].notna()

assert flight_weather.loc[
    valid_weather, "flight_temperature_c"
].notna().all(), "Missing matched temperature."

assert flight_weather.loc[
    valid_weather, "flight_wind_speed_kmh"
].notna().all(), "Missing matched wind speed."

assert flight_weather.loc[
    valid_weather, "flight_wind_direction_deg"
].notna().all(), "Missing matched wind direction."

assert (
    flight_weather.loc[valid_weather, "flight_wind_speed_kmh"] >= 0
).all(), "Negative wind speed found."

assert flight_weather.loc[
    valid_weather, "flight_wind_direction_deg"
].between(0, 360).all(), "Invalid wind direction found."

print("ALTITUDE-MATCHED WEATHER TEST: PASS")
print("Valid altitude/weather observations:", int(valid_weather.sum()))
print("Rows retained without altitude:", int((~valid_weather).sum()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Derive Wind Components and Airspeed Inputs
# MAGIC
# MAGIC Use the altitude-matched wind conditions together with aircraft ground speed and true track to derive wind-aware flight features.
# MAGIC
# MAGIC OpenSky `velocity` represents ground speed, so it will not be treated directly as true airspeed. Wind speed and direction will be used to derive the aircraft airspeed inputs required for later fuel-flow and emissions estimation.
# MAGIC
# MAGIC Rows with insufficient kinematic or weather data will be retained and flagged rather than assigned artificial values.

# COMMAND ----------

# Check inputs required for wind-aware airspeed calculation

airspeed_input_cols = [
    "velocity",
    "true_track",
    "flight_wind_speed_kmh",
    "flight_wind_direction_deg"
]

print("=== AIRSPEED INPUT CHECK ===")

for col in airspeed_input_cols:
    print(
        f"{col}:",
        "available =", flight_weather[col].notna().sum(),
        "| missing =", flight_weather[col].isna().sum()
    )

airspeed_inputs_ready = flight_weather[airspeed_input_cols].notna().all(axis=1)

print("\nTotal observations:", len(flight_weather))
print("Airspeed inputs ready:", int(airspeed_inputs_ready.sum()))
print("Airspeed inputs unavailable:", int((~airspeed_inputs_ready).sum()))

print("\nRows with unavailable inputs:")
display(
    flight_weather.loc[
        ~airspeed_inputs_ready,
        ["plane_id", "flight_id"] + airspeed_input_cols
    ].head(20)
)

# COMMAND ----------

# Derive wind components

flight_weather["velocity"] = pd.to_numeric(
    flight_weather["velocity"], errors="coerce"
)

flight_weather["true_track"] = pd.to_numeric(
    flight_weather["true_track"], errors="coerce"
)

# Convert wind speed from km/h to m/s
flight_weather["flight_wind_speed_ms"] = (
    flight_weather["flight_wind_speed_kmh"] / 3.6
)

# Relative angle between wind FROM direction and aircraft ground track
relative_wind_angle_rad = np.deg2rad(
    flight_weather["flight_wind_direction_deg"]
    - flight_weather["true_track"]
)

# Positive = headwind, Negative = tailwind
flight_weather["headwind_component_ms"] = (
    flight_weather["flight_wind_speed_ms"]
    * np.cos(relative_wind_angle_rad)
)

# Signed crosswind component
flight_weather["crosswind_component_ms"] = (
    flight_weather["flight_wind_speed_ms"]
    * np.sin(relative_wind_angle_rad)
)

print("=== WIND COMPONENTS SAMPLE ===")

display(
    flight_weather[
        [
            "plane_id",
            "velocity",
            "true_track",
            "flight_wind_speed_ms",
            "flight_wind_direction_deg",
            "headwind_component_ms",
            "crosswind_component_ms"
        ]
    ].head(10)
)

# COMMAND ----------

# Test derived wind components

valid_wind = airspeed_inputs_ready

assert flight_weather.loc[
    valid_wind, "flight_wind_speed_ms"
].ge(0).all(), "Negative wind speed found."

assert (
    flight_weather.loc[valid_wind, "headwind_component_ms"].abs()
    <= flight_weather.loc[valid_wind, "flight_wind_speed_ms"] + 1e-9
).all(), "Headwind component exceeds total wind speed."

assert (
    flight_weather.loc[valid_wind, "crosswind_component_ms"].abs()
    <= flight_weather.loc[valid_wind, "flight_wind_speed_ms"] + 1e-9
).all(), "Crosswind component exceeds total wind speed."

assert flight_weather.loc[
    valid_wind,
    ["headwind_component_ms", "crosswind_component_ms"]
].notna().all().all(), "Missing wind components found."

print("WIND COMPONENT TEST: PASS")
print("Valid wind-component observations:", int(valid_wind.sum()))
print("Rows retained without wind components:", int((~valid_wind).sum()))

# COMMAND ----------

# Derive airspeed from ground-speed and wind vectors

track_rad = np.deg2rad(flight_weather["true_track"])
wind_from_rad = np.deg2rad(flight_weather["flight_wind_direction_deg"])

# Ground-velocity vector: east and north components
ground_east_ms = flight_weather["velocity"] * np.sin(track_rad)
ground_north_ms = flight_weather["velocity"] * np.cos(track_rad)

# Meteorological wind direction tells where wind comes FROM.
# Convert it to the direction the wind moves TOWARD.
wind_east_ms = -flight_weather["flight_wind_speed_ms"] * np.sin(wind_from_rad)
wind_north_ms = -flight_weather["flight_wind_speed_ms"] * np.cos(wind_from_rad)

# Air velocity = ground velocity - wind velocity
air_east_ms = ground_east_ms - wind_east_ms
air_north_ms = ground_north_ms - wind_north_ms

flight_weather["derived_airspeed_ms"] = np.sqrt(
    air_east_ms**2 + air_north_ms**2
)

# Keep airspeed unavailable when required inputs are unavailable
flight_weather.loc[
    ~airspeed_inputs_ready, "derived_airspeed_ms"
] = np.nan

print("=== DERIVED AIRSPEED SAMPLE ===")

display(
    flight_weather[
        [
            "plane_id",
            "velocity",
            "flight_wind_speed_ms",
            "headwind_component_ms",
            "crosswind_component_ms",
            "derived_airspeed_ms"
        ]
    ].head(10)
)

print(
    "\nDerived airspeed available:",
    flight_weather["derived_airspeed_ms"].notna().sum(),
    "/",
    len(flight_weather)
)

# COMMAND ----------

# Test derived airspeed

valid_airspeed = airspeed_inputs_ready

assert flight_weather.loc[
    valid_airspeed, "derived_airspeed_ms"
].notna().all(), "Missing derived airspeed for valid inputs."

assert (
    flight_weather.loc[valid_airspeed, "derived_airspeed_ms"] > 0
).all(), "Non-positive derived airspeed found."

assert flight_weather.loc[
    ~valid_airspeed, "derived_airspeed_ms"
].isna().all(), "Airspeed was created for rows with insufficient inputs."

print("DERIVED AIRSPEED TEST: PASS")
print("Derived airspeed observations:", int(valid_airspeed.sum()))
print("Rows retained without derived airspeed:", int((~valid_airspeed).sum()))

print(
    "Derived airspeed range (m/s):",
    round(flight_weather.loc[valid_airspeed, "derived_airspeed_ms"].min(), 2),
    "->",
    round(flight_weather.loc[valid_airspeed, "derived_airspeed_ms"].max(), 2)
)

# COMMAND ----------

flight_weather[
    [
        "plane_id",
        "velocity",
        "flight_wind_speed_ms",
        "derived_airspeed_ms"
    ]
].sort_values(
    "derived_airspeed_ms",
    ascending=False
).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Prepare OpenAP Fuel-Flow Inputs
# MAGIC
# MAGIC Prepare the validated flight observations for fuel-flow estimation using OpenAP.
# MAGIC
# MAGIC Only observations with supported aircraft types and the required flight-state inputs will be eligible for fuel-flow estimation. Rows that are not eligible will be retained and explicitly flagged rather than assigned estimated values.

# COMMAND ----------

# Check readiness for OpenAP fuel-flow estimation

fuel_input_cols = [
    "openap_typecode",
    "derived_airspeed_ms",
    "flight_altitude_m",
    "vertical_rate"
]

# Ensure vertical rate is numeric
flight_weather["vertical_rate"] = pd.to_numeric(
    flight_weather["vertical_rate"],
    errors="coerce"
)

# Parse FuelFlow readiness correctly from CSV text
fuel_model_ready_bool = (
    flight_weather["fuel_model_ready"]
    .astype("string")
    .str.lower()
    .eq("true")
)

# Sanity check: retain rows, but do not send implausible airspeed to OpenAP
flight_weather["airspeed_plausible"] = (
    flight_weather["derived_airspeed_ms"].between(0, 350)
)

print("=== OPENAP FUEL-FLOW INPUT CHECK ===")

for col in fuel_input_cols:
    print(
        f"{col}:",
        "available =", flight_weather[col].notna().sum(),
        "| missing =", flight_weather[col].isna().sum()
    )

fuel_ready = (
    fuel_model_ready_bool
    & flight_weather["openap_typecode"].notna()
    & flight_weather["derived_airspeed_ms"].notna()
    & flight_weather["airspeed_plausible"]
    & flight_weather["flight_altitude_m"].notna()
    & flight_weather["vertical_rate"].notna()
)

flight_weather["fuel_flow_ready"] = fuel_ready

print("\nTotal observations:", len(flight_weather))
print("Fuel-flow ready:", int(fuel_ready.sum()))
print("Not fuel-flow ready:", int((~fuel_ready).sum()))
print(
    "Implausible airspeed rows:",
    int(
        flight_weather["derived_airspeed_ms"].notna().sum()
        - flight_weather["airspeed_plausible"].sum()
    )
)

display(
    flight_weather[
        [
            "plane_id",
            "flight_id",
            "openap_typecode",
            "fuel_model_ready",
            "derived_airspeed_ms",
            "airspeed_plausible",
            "flight_altitude_m",
            "vertical_rate",
            "fuel_flow_ready"
        ]
    ].head(10)
)

# COMMAND ----------

# MAGIC %pip install openap

# COMMAND ----------

from openap import prop

# Convert flight-state data to OpenAP input units
flight_weather["tas_kts"] = flight_weather["derived_airspeed_ms"] * 1.943844
flight_weather["altitude_ft"] = flight_weather["flight_altitude_m"] * 3.28084
flight_weather["vertical_rate_fpm"] = flight_weather["vertical_rate"] * 196.850394

# Estimate aircraft mass as 85% of MTOW
ready_codes = (
    flight_weather.loc[
        flight_weather["fuel_flow_ready"],
        "openap_typecode"
    ]
    .dropna()
    .unique()
)

mass_lookup = {}

for code in ready_codes:
    ac_info = prop.aircraft(code, use_synonym=True)
    mass_lookup[code] = ac_info["mtow"] * 0.85

flight_weather["estimated_mass_kg"] = (
    flight_weather["openap_typecode"].map(mass_lookup)
)

flight_weather.loc[
    ~flight_weather["fuel_flow_ready"],
    "estimated_mass_kg"
] = np.nan

print("=== OPENAP INPUTS PREPARED ===")
print(
    "Mass estimates available:",
    flight_weather["estimated_mass_kg"].notna().sum()
)

display(
    flight_weather.loc[
        flight_weather["fuel_flow_ready"],
        [
            "plane_id",
            "openap_typecode",
            "estimated_mass_kg",
            "tas_kts",
            "altitude_ft",
            "vertical_rate_fpm"
        ]
    ].head(10)
)

# COMMAND ----------

from openap import FuelFlow

flight_weather["fuel_flow_kg_s"] = np.nan

# Create one FuelFlow model per aircraft type
ff_cache = {}

ready_rows = flight_weather.index[
    flight_weather["fuel_flow_ready"]
]

for idx in ready_rows:
    typecode = str(
        flight_weather.at[idx, "openap_typecode"]
    ).upper()

    if typecode not in ff_cache:
        ff_cache[typecode] = FuelFlow(
            ac=typecode,
            use_synonym=True
        )

    ff = ff_cache[typecode]

    fuel_flow = ff.enroute(
        mass=flight_weather.at[idx, "estimated_mass_kg"],
        tas=flight_weather.at[idx, "tas_kts"],
        alt=flight_weather.at[idx, "altitude_ft"],
        vs=flight_weather.at[idx, "vertical_rate_fpm"]
    )

    flight_weather.at[idx, "fuel_flow_kg_s"] = fuel_flow

print("=== FUEL FLOW RESULTS ===")
print(
    "Fuel-flow estimates:",
    flight_weather["fuel_flow_kg_s"].notna().sum()
)

print(
    "FuelFlow models created:",
    len(ff_cache)
)

display(
    flight_weather.loc[
        flight_weather["fuel_flow_ready"],
        [
            "plane_id",
            "openap_typecode",
            "estimated_mass_kg",
            "tas_kts",
            "altitude_ft",
            "vertical_rate_fpm",
            "fuel_flow_kg_s"
        ]
    ].head(10)
)

# COMMAND ----------

# Test OpenAP fuel-flow results

fuel_eligible = flight_weather["fuel_flow_ready"]

flight_weather["fuel_flow_success"] = (
    fuel_eligible
    & flight_weather["fuel_flow_kg_s"].notna()
    & np.isfinite(flight_weather["fuel_flow_kg_s"])
    & (flight_weather["fuel_flow_kg_s"] > 0)
)

fuel_results = flight_weather.loc[
    flight_weather["fuel_flow_success"],
    "fuel_flow_kg_s"
]

eligible_count = int(fuel_eligible.sum())
success_count = int(flight_weather["fuel_flow_success"].sum())
failed_count = eligible_count - success_count

assert len(fuel_results) == success_count, \
    "Fuel-flow success count mismatch."

assert np.isfinite(fuel_results).all(), \
    "Non-finite successful fuel-flow values found."

assert (fuel_results > 0).all(), \
    "Non-positive successful fuel-flow values found."

print("FUEL FLOW TEST: PASS")
print("Fuel-flow eligible:", eligible_count)
print("Fuel-flow successful:", success_count)
print("Fuel-flow failed:", failed_count)

print(
    "Fuel-flow range (kg/s):",
    round(fuel_results.min(), 3),
    "->",
    round(fuel_results.max(), 3)
)

print(
    "Average fuel flow (kg/s):",
    round(fuel_results.mean(), 3)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Estimate Aircraft Emissions
# MAGIC
# MAGIC Estimate aircraft emission rates for observations with valid OpenAP fuel-flow results.
# MAGIC
# MAGIC Emissions are calculated only for eligible observations using the aircraft type, fuel-flow estimate, true airspeed, and altitude. Rows without sufficient inputs are retained with unavailable emission estimates.

# COMMAND ----------

# Calculate aircraft emission rates using OpenAP

from openap import Emission

emission_cols = [
    "co2_g_s",
    "nox_g_s",
    "co_g_s",
    "hc_g_s",
    "sox_g_s",
    "soot_g_s"
]

# Initialize emission columns
for col in emission_cols:
    flight_weather[col] = np.nan

# Create one Emission model per aircraft type
emission_cache = {}

successful_rows = flight_weather.index[
    flight_weather["fuel_flow_success"]
]

for idx in successful_rows:
    typecode = str(
        flight_weather.at[idx, "openap_typecode"]
    ).upper()

    if typecode not in emission_cache:
        emission_cache[typecode] = Emission(
            ac=typecode,
            use_synonym=True
        )

    emission = emission_cache[typecode]

    ff = flight_weather.at[idx, "fuel_flow_kg_s"]
    tas = flight_weather.at[idx, "tas_kts"]
    alt = flight_weather.at[idx, "altitude_ft"]

    flight_weather.at[idx, "co2_g_s"] = emission.co2(ff)
    flight_weather.at[idx, "nox_g_s"] = emission.nox(ff, tas, alt)
    flight_weather.at[idx, "co_g_s"] = emission.co(ff, tas, alt)
    flight_weather.at[idx, "hc_g_s"] = emission.hc(ff, tas, alt)
    flight_weather.at[idx, "sox_g_s"] = emission.sox(ff)
    flight_weather.at[idx, "soot_g_s"] = emission.soot(ff)

print("=== EMISSION RESULTS ===")
print(
    "Emission estimates:",
    flight_weather["co2_g_s"].notna().sum()
)
print(
    "Emission models created:",
    len(emission_cache)
)

display(
    flight_weather.loc[
        flight_weather["fuel_flow_success"],
        [
            "plane_id",
            "openap_typecode",
            "fuel_flow_kg_s",
            "co2_g_s",
            "nox_g_s",
            "co_g_s",
            "hc_g_s",
            "sox_g_s",
            "soot_g_s"
        ]
    ].head(10)
)

# COMMAND ----------

# Test aircraft emission results

valid_emissions = flight_weather["fuel_flow_success"]

assert flight_weather.loc[
    valid_emissions, "co2_g_s"
].notna().all(), "Missing CO2 estimates."

assert flight_weather.loc[
    valid_emissions, emission_cols
].ge(0).all().all(), "Negative emission values found."

assert flight_weather.loc[
    ~valid_emissions, emission_cols
].isna().all().all(), \
    "Emissions were calculated for rows without successful fuel flow."

assert (
    flight_weather.loc[
        valid_emissions,
        emission_cols
    ]
    .notna()
    .all()
    .all()
), "Missing emission estimates found."

print("EMISSION TEST: PASS")
print("Emission observations:", int(valid_emissions.sum()))
print(
    "Rows retained without emissions:",
    int((~valid_emissions).sum())
)

print(
    "CO2 range (g/s):",
    round(
        flight_weather.loc[
            valid_emissions,
            "co2_g_s"
        ].min(),
        3
    ),
    "->",
    round(
        flight_weather.loc[
            valid_emissions,
            "co2_g_s"
        ].max(),
        3
    )
)

print(
    "NOx range (g/s):",
    round(
        flight_weather.loc[
            valid_emissions,
            "nox_g_s"
        ].min(),
        3
    ),
    "->",
    round(
        flight_weather.loc[
            valid_emissions,
            "nox_g_s"
        ].max(),
        3
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Prepare Final Observation-Level Dataset
# MAGIC
# MAGIC Prepare an analysis-ready observation-level dataset that can be appended across repeated pipeline runs.
# MAGIC
# MAGIC Each flight observation remains a separate record. The dataset preserves timestamp, position, aircraft, weather, fuel-flow, emissions, and Saudi-boundary information required for future spatiotemporal and traffic analysis.
# MAGIC
# MAGIC No traffic corridor or route classification is created from the current single snapshot. These patterns will be derived after observations accumulate across repeated pipeline runs.

# COMMAND ----------

# Select analysis-ready observation-level columns

final_columns = [
    # Flight identity and time
    "batch_id",
    "plane_id",
    "flight_id",
    "origin_country",
    "time_position",
    "last_contact",

    # Position and movement
    "longitude",
    "latitude",
    "flight_altitude_m",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "inside_saudi_boundary",

    # Aircraft information
    "registration",
    "manufacturername",
    "model",
    "typecode",
    "openap_typecode",
    "aircraft_metadata_matched",

    # Matched weather
    "weather_batch_available",
    "weather_time",
    "weather_location_id",
    "weather_latitude",
    "weather_longitude",
    "weather_distance_km",
    "pressure_level_hpa",
    "flight_temperature_c",
    "flight_wind_speed_kmh",
    "flight_wind_direction_deg",

    # Derived flight features
    "headwind_component_ms",
    "crosswind_component_ms",
    "derived_airspeed_ms",
    "airspeed_plausible",

    # Fuel estimation
    "fuel_flow_ready",
    "fuel_flow_success",
    "estimated_mass_kg",
    "fuel_flow_kg_s",

    # Emission estimates
    "co2_g_s",
    "nox_g_s",
    "co_g_s",
    "hc_g_s",
    "sox_g_s",
    "soot_g_s"
]

final_df = flight_weather[final_columns].copy()

print("=== FINAL DATASET STRUCTURE ===")
print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))
print("Duplicate column names:", final_df.columns.duplicated().sum())

display(final_df.head())

# COMMAND ----------

# Audit missing values in the final dataset

null_audit = pd.DataFrame({
    "missing_count": final_df.isna().sum(),
    "missing_percent": (final_df.isna().mean() * 100).round(1)
})

null_audit = (
    null_audit[
        null_audit["missing_count"] > 0
    ]
    .sort_values("missing_count", ascending=False)
    .reset_index()
    .rename(columns={"index": "column"})
)

print("=== FINAL DATASET NULL AUDIT ===")
display(null_audit)

# COMMAND ----------

# Check Saudi-boundary distribution

inside_saudi_bool = (
    final_df["inside_saudi_boundary"]
    .astype("string")
    .str.lower()
    .eq("true")
)

fuel_ready_bool = (
    final_df["fuel_flow_ready"]
    .astype("string")
    .str.lower()
    .eq("true")
)

print("=== SAUDI BOUNDARY CHECK ===")

print(
    inside_saudi_bool.value_counts(dropna=False)
)

print(
    "\nFuel/emission-ready observations inside Saudi boundary:",
    int((inside_saudi_bool & fuel_ready_bool).sum())
)

# COMMAND ----------

# Test final observation-level dataset

inside_saudi_bool = (
    final_df["inside_saudi_boundary"]
    .astype("string")
    .str.lower()
    .eq("true")
)

fuel_success_bool = (
    final_df["fuel_flow_success"]
    .astype("string")
    .str.lower()
    .eq("true")
)

assert len(final_df) == len(flight_weather), \
    "Final dataset changed the observation grain."

assert final_df["plane_id"].notna().all(), \
    "Missing plane_id found."

assert final_df["time_position"].notna().all(), \
    "Missing observation timestamp found."

assert final_df["latitude"].notna().all(), \
    "Missing latitude found."

assert final_df["longitude"].notna().all(), \
    "Missing longitude found."

assert final_df["inside_saudi_boundary"].notna().all(), \
    "Missing Saudi-boundary flag found."

assert final_df.loc[
    fuel_success_bool,
    "fuel_flow_kg_s"
].notna().all(), \
    "Successful fuel-flow observations contain missing estimates."

print("FINAL DATASET TEST: PASS")

print("Observations retained:", len(final_df))

print(
    "Inside Saudi boundary:",
    int(inside_saudi_bool.sum())
)

print(
    "Inside Saudi with successful fuel/emission estimates:",
    int(
        (
            inside_saudi_bool
            & fuel_success_bool
        ).sum()
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Transformation Rules and Final Output
# MAGIC
# MAGIC Document the transformation rules implemented in Task 4 and verify the final analysis-ready dataset before export.
# MAGIC
# MAGIC The rules describe the joins, weather matching, derived flight features, fuel-flow estimation, emission estimation, and preservation of Saudi airspace membership.
# MAGIC
# MAGIC Final quality checks are performed before exporting the observation-level dataset to the Gold layer in Azure Data Lake Storage.

# COMMAND ----------

# Transformation rules implemented in Task 4

rules = pd.DataFrame([
    {
        "Rule ID": "R1",
        "Description": "Join flight observations with aircraft metadata using a left join.",
        "Input Column(s)": "plane_id",
        "Output Column": "aircraft metadata columns"
    },
    {
        "Rule ID": "R2",
        "Description": "Match each flight observation to the nearest available weather location within the same batch, then match to the rounded weather observation hour.",
        "Input Column(s)": "batch_id, latitude, longitude, time_position",
        "Output Column": "weather_location_id, weather_time, weather_distance_km"
    },
    {
        "Rule ID": "R3",
        "Description": "Select the atmospheric pressure level closest to the aircraft altitude.",
        "Input Column(s)": "flight_altitude_m, geopotential height columns",
        "Output Column": "pressure_level_hpa"
    },
    {
        "Rule ID": "R4",
        "Description": "Select temperature and wind conditions from the altitude-matched pressure level.",
        "Input Column(s)": "pressure_level_hpa, pressure-level weather columns",
        "Output Column": "flight_temperature_c, flight_wind_speed_kmh, flight_wind_direction_deg"
    },
    {
        "Rule ID": "R5",
        "Description": "Derive wind components and air-relative speed from ground velocity, track, and matched wind.",
        "Input Column(s)": "velocity, true_track, flight_wind_speed_kmh, flight_wind_direction_deg",
        "Output Column": "headwind_component_ms, crosswind_component_ms, derived_airspeed_ms"
    },
    {
        "Rule ID": "R6",
        "Description": "Prepare supported observations for OpenAP fuel-flow estimation, apply an airspeed plausibility check, and estimate aircraft mass as 85% of MTOW.",
        "Input Column(s)": "openap_typecode, derived_airspeed_ms, flight_altitude_m, vertical_rate",
        "Output Column": "airspeed_plausible, fuel_flow_ready, estimated_mass_kg"
    },
    {
        "Rule ID": "R7",
        "Description": "Estimate instantaneous aircraft fuel-flow rate using OpenAP and flag successful calculations.",
        "Input Column(s)": "openap_typecode, estimated_mass_kg, tas_kts, altitude_ft, vertical_rate_fpm",
        "Output Column": "fuel_flow_kg_s, fuel_flow_success"
    },
    {
        "Rule ID": "R8",
        "Description": "Estimate instantaneous aircraft emission rates using successful OpenAP fuel-flow results.",
        "Input Column(s)": "openap_typecode, fuel_flow_kg_s, tas_kts, altitude_ft",
        "Output Column": "co2_g_s, nox_g_s, co_g_s, hc_g_s, sox_g_s, soot_g_s"
    },
    {
        "Rule ID": "R9",
        "Description": "Preserve Saudi-boundary membership for downstream Saudi airspace analysis.",
        "Input Column(s)": "inside_saudi_boundary",
        "Output Column": "inside_saudi_boundary"
    }
])

display(rules)

# COMMAND ----------

# Edge-case tests required for Task 4

fuel_ready_bool = (
    final_df["fuel_flow_ready"]
    .astype("string")
    .str.lower()
    .eq("true")
)

fuel_success_bool = (
    final_df["fuel_flow_success"]
    .astype("string")
    .str.lower()
    .eq("true")
)

# 1. Empty input should remain empty
empty_input = final_df.iloc[0:0].copy()
assert empty_input.empty, "Empty-input test failed."

# 2. Core observation keys must not be null
assert final_df["plane_id"].notna().all(), \
    "Null plane_id found."

assert final_df["time_position"].notna().all(), \
    "Null time_position found."

# 3. Missing flight_id is allowed and must not remove the observation
assert len(final_df) == len(flight_weather), \
    "Final dataset changed the observation grain."

# 4. Ineligible fuel rows must retain missing estimates
assert final_df.loc[
    ~fuel_ready_bool,
    "fuel_flow_kg_s"
].isna().all(), \
    "Fuel flow found for an ineligible observation."

# 5. Failed OpenAP calculations must also remain missing
assert final_df.loc[
    fuel_ready_bool & ~fuel_success_bool,
    "fuel_flow_kg_s"
].isna().all(), \
    "Failed fuel-flow calculation contains an estimate."

print("EDGE-CASE TESTS: PASS")
print("Empty input handled correctly.")
print("Null core keys checked.")
print("Optional missing flight_id retained.")
print("Ineligible fuel rows handled correctly.")
print("Failed OpenAP fuel-flow rows handled correctly.")

# COMMAND ----------

# Final quality checks before export

fuel_ready_bool = (
    final_df["fuel_flow_ready"]
    .astype("string")
    .str.lower()
    .eq("true")
)

fuel_success_bool = (
    final_df["fuel_flow_success"]
    .astype("string")
    .str.lower()
    .eq("true")
)

inside_saudi_bool = (
    final_df["inside_saudi_boundary"]
    .astype("string")
    .str.lower()
    .eq("true")
)

assert len(final_df) == len(flight_weather), \
    "Final dataset changed the observation grain."

assert final_df.columns.duplicated().sum() == 0, \
    "Duplicate column names found."

assert final_df["plane_id"].notna().all(), \
    "Missing plane_id found."

assert final_df["time_position"].notna().all(), \
    "Missing time_position found."

assert final_df[["latitude", "longitude"]].notna().all().all(), \
    "Missing flight coordinates found."

assert final_df["inside_saudi_boundary"].notna().all(), \
    "Missing Saudi-boundary flag found."

emission_cols = [
    "co2_g_s", "nox_g_s", "co_g_s",
    "hc_g_s", "sox_g_s", "soot_g_s"
]

# Successful fuel-flow rows must have emissions
assert final_df.loc[
    fuel_success_bool, emission_cols
].notna().all().all(), \
    "Missing emissions found for successful fuel-flow observations."

# Rows without successful fuel flow must not have emissions
assert final_df.loc[
    ~fuel_success_bool, emission_cols
].isna().all().all(), \
    "Emissions found without successful fuel flow."

print("FINAL QUALITY CHECKS: PASS")
print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))
print("Fuel-flow ready:", int(fuel_ready_bool.sum()))
print("Fuel-flow successful:", int(fuel_success_bool.sum()))
print("Inside Saudi boundary:", int(inside_saudi_bool.sum()))


# COMMAND ----------

# Export the final analysis-ready dataset to Gold ADLS

final_path = (
    "abfss://gold@skyprint74815.dfs.core.windows.net/"
    "skyprint/final.csv"
)

csv_text = final_df.to_csv(index=False)

dbutils.fs.put(
    final_path,
    csv_text,
    overwrite=True
)

print("=== FINAL DATASET EXPORTED ===")
print("Path:", final_path)
print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))

# COMMAND ----------

# Verify the exported final dataset

saved_final = (
    spark.read
    .option("header", "true")
    .csv(final_path)
)

assert saved_final.count() == len(final_df), \
    "Saved file row count does not match final_df."

assert saved_final.columns == list(final_df.columns), \
    "Saved file columns do not match final_df."

assert saved_final.filter("plane_id IS NULL").count() == 0, \
    "Saved file contains missing plane_id values."

assert saved_final.filter("time_position IS NULL").count() == 0, \
    "Saved file contains missing time_position values."

print("FINAL EXPORT VERIFICATION: PASS")
print("Saved rows:", saved_final.count())
print("Saved columns:", len(saved_final.columns))
print("File:", final_path)

# COMMAND ----------

# Test left-join behavior with a null key

left_test = pd.DataFrame({
    "plane_id": ["ABC123", None],
    "observation": ["matched_row", "null_key_row"]
})

right_test = pd.DataFrame({
    "plane_id": ["ABC123"],
    "aircraft_type": ["TEST_TYPE"]
})

result_test = left_test.merge(
    right_test,
    on="plane_id",
    how="left",
    validate="many_to_one"
)

# The left join must preserve both observations
assert len(result_test) == 2, \
    "Left join unexpectedly removed an observation."

# The valid key must match
assert result_test.loc[
    result_test["plane_id"] == "ABC123",
    "aircraft_type"
].iloc[0] == "TEST_TYPE", \
    "Valid key did not match correctly."

# The null key must be retained without a false match
null_row = result_test[result_test["plane_id"].isna()]

assert len(null_row) == 1, \
    "Null-key observation was not retained."

assert null_row["aircraft_type"].isna().all(), \
    "Null key created an unexpected match."

print("NULL-KEY JOIN TEST: PASS")
print("Valid key matched correctly.")
print("Null-key observation retained without a false match.")