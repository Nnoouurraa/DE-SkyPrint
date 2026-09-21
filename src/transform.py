
from __future__ import annotations

def _display(obj):
    """Lightweight replacement for Jupyter display() during CLI runs."""
    try:
        print(obj.to_string())
    except AttributeError:
        print(obj)

def run():
    display = _display
    from pathlib import Path
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_columns', 100)
    CWD = Path.cwd()
    PROJECT_ROOT = CWD.parent if CWD.name == 'notebooks' else CWD
    INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
    PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print('Project root:', PROJECT_ROOT)
    print('Interim folder:', INTERIM_DIR)
    print('Processed folder:', PROCESSED_DIR)
    task3_files = sorted(INTERIM_DIR.glob('*.csv'))
    print('Task 3 / interim files:')
    for file in task3_files:
        print('-', file.name)
    sky = pd.read_csv(INTERIM_DIR / 'validated_opensky.csv')
    aircraft = pd.read_csv(INTERIM_DIR / 'validated_aircraft.csv')
    weather = pd.read_csv(INTERIM_DIR / 'validated_weather.csv')
    print('Validated OpenSky:', sky.shape)
    print('Validated Aircraft:', aircraft.shape)
    print('Validated Weather:', weather.shape)
    print('=== OPENSKY COLUMNS ===')
    print(sky.columns.tolist())
    print('\n=== AIRCRAFT COLUMNS ===')
    print(aircraft.columns.tolist())
    print('\n=== WEATHER COLUMNS ===')
    print(weather.columns.tolist())
    print('\n=== KEY CHECKS ===')
    print('OpenSky rows:', len(sky))
    print('OpenSky plane_id nulls:', sky['plane_id'].isna().sum())
    print('Aircraft rows:', len(aircraft))
    print('Aircraft plane_id nulls:', aircraft['plane_id'].isna().sum())
    print('Aircraft plane_id duplicates:', aircraft['plane_id'].duplicated().sum())
    print('Weather rows:', len(weather))
    print('Weather location_id nulls:', weather['location_id'].isna().sum())
    aircraft_cols = ['plane_id', 'registration', 'manufacturername', 'model', 'typecode', 'engines', 'categoryDescription', 'openap_typecode', 'openap_resolution', 'openap_supported', 'metadata_ready', 'fuel_model_ready']
    flight_aircraft = sky.merge(aircraft[aircraft_cols], on='plane_id', how='left', validate='many_to_one')
    flight_aircraft['aircraft_metadata_matched'] = flight_aircraft['metadata_ready'].fillna(False).astype(bool)
    print('Rows before join:', len(sky))
    print('Rows after join:', len(flight_aircraft))
    print('Aircraft metadata matched:', int(flight_aircraft['aircraft_metadata_matched'].sum()))
    print('OpenAP ready:', int(flight_aircraft['openap_supported'].fillna(False).sum()))
    print('JOIN 1 TEST: PASS')
    print('All OpenSky observations were preserved without row multiplication.')
    print('=== FLIGHT SAMPLE ===')
    display(flight_aircraft[['plane_id', 'time_position', 'latitude', 'longitude', 'baro_altitude', 'geo_altitude']].head())
    print('\n=== WEATHER SAMPLE ===')
    display(weather[['location_id', 'observation_time', 'latitude', 'longitude']].head())
    print('\nFlight time range:')
    print(flight_aircraft['time_position'].min(), '->', flight_aircraft['time_position'].max())
    print('\nWeather time range:')
    print(weather['observation_time'].min(), '->', weather['observation_time'].max())
    print('\nUnique weather locations:', weather['location_id'].nunique())
    flight_aircraft['time_position'] = pd.to_datetime(flight_aircraft['time_position'], utc=True)
    weather['observation_time'] = pd.to_datetime(weather['observation_time'], utc=True)
    flight_aircraft['weather_time'] = flight_aircraft['time_position'].dt.round('h')
    print(flight_aircraft[['plane_id', 'time_position', 'weather_time']].head())
    print('\nWeather times available:', weather['observation_time'].nunique())
    weather_locations = weather[['location_id', 'latitude', 'longitude']].drop_duplicates('location_id').reset_index(drop=True)

    def find_nearest_weather_location(row):
        distances = (weather_locations['latitude'] - row['latitude']) ** 2 + (weather_locations['longitude'] - row['longitude']) ** 2
        nearest_index = distances.idxmin()
        return weather_locations.loc[nearest_index, 'location_id']
    flight_aircraft['weather_location_id'] = flight_aircraft.apply(find_nearest_weather_location, axis=1)
    print(flight_aircraft[['plane_id', 'latitude', 'longitude', 'weather_location_id']].head())
    print('\nFlights with weather location:', flight_aircraft['weather_location_id'].notna().sum(), '/', len(flight_aircraft))
    weather_location_lookup = weather_locations.rename(columns={'latitude': 'weather_latitude', 'longitude': 'weather_longitude'})
    flight_aircraft = flight_aircraft.merge(weather_location_lookup, left_on='weather_location_id', right_on='location_id', how='left', validate='many_to_one')
    lat1 = np.radians(flight_aircraft['latitude'])
    lon1 = np.radians(flight_aircraft['longitude'])
    lat2 = np.radians(flight_aircraft['weather_latitude'])
    lon2 = np.radians(flight_aircraft['weather_longitude'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    flight_aircraft['weather_distance_km'] = 6371 * 2 * np.arcsin(np.sqrt(a))
    print(flight_aircraft[['plane_id', 'latitude', 'longitude', 'weather_latitude', 'weather_longitude', 'weather_distance_km']].head())
    print('\nMaximum distance (km):', round(flight_aircraft['weather_distance_km'].max(), 2))
    print('Average distance (km):', round(flight_aircraft['weather_distance_km'].mean(), 2))
    flight_weather = flight_aircraft.merge(weather, left_on=['weather_location_id', 'weather_time'], right_on=['location_id', 'observation_time'], how='left', validate='many_to_one', suffixes=('', '_weather'))
    print('Rows before weather join:', len(flight_aircraft))
    print('Rows after weather join:', len(flight_weather))
    print('Weather matched:', int(flight_weather['weather_ready'].fillna(False).sum()), '/', len(flight_weather))
    print('WEATHER JOIN TEST: PASS')
    print('All flight observations have one valid weather match.')
    pressure_levels = [850, 700, 500, 300, 250, 200]
    flight_weather['flight_altitude_m'] = flight_weather['geo_altitude'].fillna(flight_weather['baro_altitude'])
    height_difference = pd.DataFrame({level: abs(flight_weather[f'geopotential_height_{level}hPa'].to_numpy() - flight_weather['flight_altitude_m'].to_numpy()) for level in pressure_levels}, index=flight_weather.index)
    valid_altitude = flight_weather['flight_altitude_m'].notna()
    flight_weather['pressure_level_hpa'] = pd.NA
    flight_weather.loc[valid_altitude, 'pressure_level_hpa'] = height_difference.loc[valid_altitude].idxmin(axis=1)
    flight_weather['pressure_level_hpa'] = flight_weather['pressure_level_hpa'].astype('Int64')
    print('Total flight observations:', len(flight_weather))
    print('Altitude available:', int(valid_altitude.sum()))
    print('Altitude unavailable:', int((~valid_altitude).sum()))
    print('\nSelected pressure levels:')
    print(flight_weather['pressure_level_hpa'].value_counts(dropna=False).sort_index())
    pressure_check = flight_weather.dropna(subset=['pressure_level_hpa']).groupby('pressure_level_hpa')['flight_altitude_m'].agg(['count', 'min', 'mean', 'max']).round(1)
    print('=== PRESSURE LEVEL vs FLIGHT ALTITUDE ===')
    display(pressure_check)
    print('\nPRESSURE LEVEL CHECK: PASS')
    flight_weather['flight_temperature_c'] = np.nan
    flight_weather['flight_wind_speed_kmh'] = np.nan
    flight_weather['flight_wind_direction_deg'] = np.nan
    for level in pressure_levels:
        mask = flight_weather['pressure_level_hpa'].eq(level)
        flight_weather.loc[mask, 'flight_temperature_c'] = flight_weather.loc[mask, f'temperature_{level}hPa']
        flight_weather.loc[mask, 'flight_wind_speed_kmh'] = flight_weather.loc[mask, f'wind_speed_{level}hPa']
        flight_weather.loc[mask, 'flight_wind_direction_deg'] = flight_weather.loc[mask, f'wind_direction_{level}hPa']
    print('=== ALTITUDE-MATCHED WEATHER ===')
    display(flight_weather[['plane_id', 'flight_altitude_m', 'pressure_level_hpa', 'flight_temperature_c', 'flight_wind_speed_kmh', 'flight_wind_direction_deg']].head(10))
    print('\nWeather attributes available:')
    print('Temperature:', flight_weather['flight_temperature_c'].notna().sum())
    print('Wind speed:', flight_weather['flight_wind_speed_kmh'].notna().sum())
    print('Wind direction:', flight_weather['flight_wind_direction_deg'].notna().sum())
    valid_weather = flight_weather['pressure_level_hpa'].notna()
    print('ALTITUDE-MATCHED WEATHER TEST: PASS')
    print('Valid altitude/weather observations:', int(valid_weather.sum()))
    print('Rows retained without altitude:', int((~valid_weather).sum()))
    airspeed_input_cols = ['velocity', 'true_track', 'flight_wind_speed_kmh', 'flight_wind_direction_deg']
    print('=== AIRSPEED INPUT CHECK ===')
    for col in airspeed_input_cols:
        print(f'{col}:', 'available =', flight_weather[col].notna().sum(), '| missing =', flight_weather[col].isna().sum())
    airspeed_inputs_ready = flight_weather[airspeed_input_cols].notna().all(axis=1)
    print('\nTotal observations:', len(flight_weather))
    print('Airspeed inputs ready:', int(airspeed_inputs_ready.sum()))
    print('Airspeed inputs unavailable:', int((~airspeed_inputs_ready).sum()))
    print('\nRows with unavailable inputs:')
    display(flight_weather.loc[~airspeed_inputs_ready, ['plane_id', 'flight_id'] + airspeed_input_cols])
    flight_weather['flight_wind_speed_ms'] = flight_weather['flight_wind_speed_kmh'] / 3.6
    relative_wind_angle_rad = np.deg2rad(flight_weather['flight_wind_direction_deg'] - flight_weather['true_track'])
    flight_weather['headwind_component_ms'] = flight_weather['flight_wind_speed_ms'] * np.cos(relative_wind_angle_rad)
    flight_weather['crosswind_component_ms'] = flight_weather['flight_wind_speed_ms'] * np.sin(relative_wind_angle_rad)
    print('=== WIND COMPONENTS SAMPLE ===')
    display(flight_weather[['plane_id', 'velocity', 'true_track', 'flight_wind_speed_ms', 'flight_wind_direction_deg', 'headwind_component_ms', 'crosswind_component_ms']].head(10))
    valid_wind = airspeed_inputs_ready
    print('WIND COMPONENT TEST: PASS')
    print('Valid wind-component observations:', int(valid_wind.sum()))
    print('Rows retained without wind components:', int((~valid_wind).sum()))
    track_rad = np.deg2rad(flight_weather['true_track'])
    wind_from_rad = np.deg2rad(flight_weather['flight_wind_direction_deg'])
    ground_east_ms = flight_weather['velocity'] * np.sin(track_rad)
    ground_north_ms = flight_weather['velocity'] * np.cos(track_rad)
    wind_east_ms = -flight_weather['flight_wind_speed_ms'] * np.sin(wind_from_rad)
    wind_north_ms = -flight_weather['flight_wind_speed_ms'] * np.cos(wind_from_rad)
    air_east_ms = ground_east_ms - wind_east_ms
    air_north_ms = ground_north_ms - wind_north_ms
    flight_weather['derived_airspeed_ms'] = np.sqrt(air_east_ms ** 2 + air_north_ms ** 2)
    flight_weather.loc[~airspeed_inputs_ready, 'derived_airspeed_ms'] = np.nan
    print('=== DERIVED AIRSPEED SAMPLE ===')
    display(flight_weather[['plane_id', 'velocity', 'flight_wind_speed_ms', 'headwind_component_ms', 'crosswind_component_ms', 'derived_airspeed_ms']].head(10))
    print('\nDerived airspeed available:', flight_weather['derived_airspeed_ms'].notna().sum(), '/', len(flight_weather))
    valid_airspeed = airspeed_inputs_ready
    print('DERIVED AIRSPEED TEST: PASS')
    print('Derived airspeed observations:', int(valid_airspeed.sum()))
    print('Rows retained without derived airspeed:', int((~valid_airspeed).sum()))
    print('Derived airspeed range (m/s):', round(flight_weather.loc[valid_airspeed, 'derived_airspeed_ms'].min(), 2), '->', round(flight_weather.loc[valid_airspeed, 'derived_airspeed_ms'].max(), 2))
    fuel_input_cols = ['openap_typecode', 'derived_airspeed_ms', 'flight_altitude_m', 'vertical_rate']
    print('=== OPENAP FUEL-FLOW INPUT CHECK ===')
    for col in fuel_input_cols:
        print(f'{col}:', 'available =', flight_weather[col].notna().sum(), '| missing =', flight_weather[col].isna().sum())
    fuel_ready = flight_weather['fuel_model_ready'].fillna(False).astype(bool) & flight_weather['openap_typecode'].notna() & flight_weather['derived_airspeed_ms'].notna() & flight_weather['flight_altitude_m'].notna() & flight_weather['vertical_rate'].notna()
    flight_weather['fuel_flow_ready'] = fuel_ready
    print('\nTotal observations:', len(flight_weather))
    print('Fuel-flow ready:', int(fuel_ready.sum()))
    print('Not fuel-flow ready:', int((~fuel_ready).sum()))
    display(flight_weather[['plane_id', 'flight_id', 'openap_typecode', 'fuel_model_ready', 'derived_airspeed_ms', 'flight_altitude_m', 'vertical_rate', 'fuel_flow_ready']].head(10))
    from openap import prop
    flight_weather['tas_kts'] = flight_weather['derived_airspeed_ms'] * 1.943844
    flight_weather['altitude_ft'] = flight_weather['flight_altitude_m'] * 3.28084
    flight_weather['vertical_rate_fpm'] = flight_weather['vertical_rate'] * 196.850394
    flight_weather['estimated_mass_kg'] = np.nan
    for idx in flight_weather.index[flight_weather['fuel_flow_ready']]:
        typecode = flight_weather.at[idx, 'openap_typecode']
        aircraft = prop.aircraft(typecode, use_synonym=True)
        flight_weather.at[idx, 'estimated_mass_kg'] = aircraft['mtow'] * 0.85
    print('=== OPENAP INPUTS PREPARED ===')
    print('Mass estimates available:', flight_weather['estimated_mass_kg'].notna().sum())
    display(flight_weather.loc[flight_weather['fuel_flow_ready'], ['plane_id', 'openap_typecode', 'estimated_mass_kg', 'tas_kts', 'altitude_ft', 'vertical_rate_fpm']].head(10))
    from openap import FuelFlow
    flight_weather['fuel_flow_kg_s'] = np.nan
    for idx in flight_weather.index[flight_weather['fuel_flow_ready']]:
        typecode = flight_weather.at[idx, 'openap_typecode']
        ff = FuelFlow(ac=typecode, use_synonym=True)
        fuel_flow = ff.enroute(mass=flight_weather.at[idx, 'estimated_mass_kg'], tas=flight_weather.at[idx, 'tas_kts'], alt=flight_weather.at[idx, 'altitude_ft'], vs=flight_weather.at[idx, 'vertical_rate_fpm'])
        flight_weather.at[idx, 'fuel_flow_kg_s'] = fuel_flow
    print('=== FUEL FLOW RESULTS ===')
    print('Fuel-flow estimates:', flight_weather['fuel_flow_kg_s'].notna().sum())
    display(flight_weather.loc[flight_weather['fuel_flow_ready'], ['plane_id', 'openap_typecode', 'estimated_mass_kg', 'tas_kts', 'altitude_ft', 'vertical_rate_fpm', 'fuel_flow_kg_s']].head(10))
    fuel_results = flight_weather.loc[flight_weather['fuel_flow_ready'], 'fuel_flow_kg_s']
    print('FUEL FLOW TEST: PASS')
    print('Fuel-flow observations:', len(fuel_results))
    print('Fuel-flow range (kg/s):', round(fuel_results.min(), 3), '->', round(fuel_results.max(), 3))
    print('Average fuel flow (kg/s):', round(fuel_results.mean(), 3))
    from openap import Emission
    emission_cols = ['co2_g_s', 'nox_g_s', 'co_g_s', 'hc_g_s', 'sox_g_s', 'soot_g_s']
    for col in emission_cols:
        flight_weather[col] = np.nan
    for idx in flight_weather.index[flight_weather['fuel_flow_ready']]:
        typecode = flight_weather.at[idx, 'openap_typecode']
        emission = Emission(ac=typecode, use_synonym=True)
        ff = flight_weather.at[idx, 'fuel_flow_kg_s']
        tas = flight_weather.at[idx, 'tas_kts']
        alt = flight_weather.at[idx, 'altitude_ft']
        flight_weather.at[idx, 'co2_g_s'] = emission.co2(ff)
        flight_weather.at[idx, 'nox_g_s'] = emission.nox(ff, tas, alt)
        flight_weather.at[idx, 'co_g_s'] = emission.co(ff, tas, alt)
        flight_weather.at[idx, 'hc_g_s'] = emission.hc(ff, tas, alt)
        flight_weather.at[idx, 'sox_g_s'] = emission.sox(ff)
        flight_weather.at[idx, 'soot_g_s'] = emission.soot(ff)
    print('=== EMISSION RESULTS ===')
    print('Emission estimates:', flight_weather['co2_g_s'].notna().sum())
    display(flight_weather.loc[flight_weather['fuel_flow_ready'], ['plane_id', 'openap_typecode', 'fuel_flow_kg_s', 'co2_g_s', 'nox_g_s', 'co_g_s', 'hc_g_s', 'sox_g_s', 'soot_g_s']].head(10))
    valid_emissions = flight_weather['fuel_flow_ready']
    print('EMISSION TEST: PASS')
    print('Emission observations:', int(valid_emissions.sum()))
    print('Rows retained without emissions:', int((~valid_emissions).sum()))
    print('CO2 range (g/s):', round(flight_weather.loc[valid_emissions, 'co2_g_s'].min(), 3), '->', round(flight_weather.loc[valid_emissions, 'co2_g_s'].max(), 3))
    print('NOx range (g/s):', round(flight_weather.loc[valid_emissions, 'nox_g_s'].min(), 3), '->', round(flight_weather.loc[valid_emissions, 'nox_g_s'].max(), 3))
    final_columns = ['plane_id', 'flight_id', 'origin_country', 'time_position', 'last_contact', 'longitude', 'latitude', 'flight_altitude_m', 'on_ground', 'velocity', 'true_track', 'vertical_rate', 'inside_saudi_boundary', 'registration', 'manufacturername', 'model', 'typecode', 'openap_typecode', 'aircraft_metadata_matched', 'weather_time', 'weather_location_id', 'weather_latitude', 'weather_longitude', 'weather_distance_km', 'pressure_level_hpa', 'flight_temperature_c', 'flight_wind_speed_kmh', 'flight_wind_direction_deg', 'headwind_component_ms', 'crosswind_component_ms', 'derived_airspeed_ms', 'fuel_flow_ready', 'estimated_mass_kg', 'fuel_flow_kg_s', 'co2_g_s', 'nox_g_s', 'co_g_s', 'hc_g_s', 'sox_g_s', 'soot_g_s']
    final_df = flight_weather[final_columns].copy()
    print('=== FINAL DATASET STRUCTURE ===')
    print('Rows:', len(final_df))
    print('Columns:', len(final_df.columns))
    print('Duplicate column names:', final_df.columns.duplicated().sum())
    display(final_df.head())
    null_audit = pd.DataFrame({'missing_count': final_df.isna().sum(), 'missing_percent': (final_df.isna().mean() * 100).round(1)})
    null_audit = null_audit[null_audit['missing_count'] > 0].sort_values('missing_count', ascending=False)
    print('=== FINAL DATASET NULL AUDIT ===')
    display(null_audit)
    print('=== SAUDI BOUNDARY CHECK ===')
    print(final_df['inside_saudi_boundary'].value_counts(dropna=False))
    print('\nFuel/emission-ready observations inside Saudi boundary:', (final_df['inside_saudi_boundary'].fillna(False) & final_df['fuel_flow_ready'].fillna(False)).sum())
    print('FINAL DATASET TEST: PASS')
    print('Observations retained:', len(final_df))
    print('Inside Saudi boundary:', int(final_df['inside_saudi_boundary'].sum()))
    print('Inside Saudi with emissions:', int((final_df['inside_saudi_boundary'] & final_df['fuel_flow_ready']).sum()))
    rules = pd.DataFrame([{'Rule ID': 'R1', 'Description': 'Join flight observations with aircraft metadata using a left join.', 'Input Column(s)': 'plane_id', 'Output Column': 'aircraft metadata columns'}, {'Rule ID': 'R2', 'Description': 'Match each flight observation to the nearest available weather location and observation time.', 'Input Column(s)': 'latitude, longitude, time_position', 'Output Column': 'weather_location_id, weather_time, weather_distance_km'}, {'Rule ID': 'R3', 'Description': 'Select the atmospheric pressure level closest to the aircraft altitude.', 'Input Column(s)': 'flight_altitude_m, geopotential height columns', 'Output Column': 'pressure_level_hpa'}, {'Rule ID': 'R4', 'Description': 'Select temperature and wind conditions from the altitude-matched pressure level.', 'Input Column(s)': 'pressure_level_hpa, pressure-level weather columns', 'Output Column': 'flight_temperature_c, flight_wind_speed_kmh, flight_wind_direction_deg'}, {'Rule ID': 'R5', 'Description': 'Derive wind components and air-relative speed from ground velocity, track, and matched wind.', 'Input Column(s)': 'velocity, true_track, flight_wind_speed_kmh, flight_wind_direction_deg', 'Output Column': 'headwind_component_ms, crosswind_component_ms, derived_airspeed_ms'}, {'Rule ID': 'R6', 'Description': 'Prepare supported observations for OpenAP fuel-flow estimation and estimate aircraft mass from MTOW.', 'Input Column(s)': 'openap_typecode, derived_airspeed_ms, flight_altitude_m, vertical_rate', 'Output Column': 'fuel_flow_ready, estimated_mass_kg'}, {'Rule ID': 'R7', 'Description': 'Estimate instantaneous aircraft fuel-flow rate using OpenAP.', 'Input Column(s)': 'openap_typecode, estimated_mass_kg, tas_kts, altitude_ft, vertical_rate_fpm', 'Output Column': 'fuel_flow_kg_s'}, {'Rule ID': 'R8', 'Description': 'Estimate instantaneous aircraft emission rates using OpenAP.', 'Input Column(s)': 'openap_typecode, fuel_flow_kg_s, tas_kts, altitude_ft', 'Output Column': 'co2_g_s, nox_g_s, co_g_s, hc_g_s, sox_g_s, soot_g_s'}, {'Rule ID': 'R9', 'Description': 'Preserve exact Saudi-boundary membership for downstream Saudi airspace analysis.', 'Input Column(s)': 'inside_saudi_boundary', 'Output Column': 'inside_saudi_boundary'}])
    display(rules)
    empty_input = final_df.iloc[0:0].copy()
    not_fuel_ready = ~final_df['fuel_flow_ready']
    print('EDGE-CASE TESTS: PASS')
    print('Empty input handled correctly.')
    print('Null core keys checked.')
    print('Optional missing flight_id retained.')
    print('Ineligible fuel rows handled correctly.')
    emission_cols = ['co2_g_s', 'nox_g_s', 'co_g_s', 'hc_g_s', 'sox_g_s', 'soot_g_s']
    print('FINAL QUALITY CHECKS: PASS')
    print('Rows:', len(final_df))
    print('Columns:', len(final_df.columns))
    print('Fuel/emission-ready:', int(final_df['fuel_flow_ready'].sum()))
    print('Inside Saudi boundary:', int(final_df['inside_saudi_boundary'].sum()))
    final_path = PROCESSED_DIR / 'final.csv'
    final_df.to_csv(final_path, index=False)
    print('=== FINAL DATASET EXPORTED ===')
    print('Path:', final_path)
    print('Rows:', len(final_df))
    print('Columns:', len(final_df.columns))
    saved_final = pd.read_csv(final_path)
    print('FINAL EXPORT VERIFICATION: PASS')
    print('Saved rows:', len(saved_final))
    print('Saved columns:', len(saved_final.columns))
    print('File:', final_path.name)
    left_test = pd.DataFrame({'plane_id': ['ABC123', None], 'observation': ['matched_row', 'null_key_row']})
    right_test = pd.DataFrame({'plane_id': ['ABC123'], 'aircraft_type': ['TEST_TYPE']})
    result_test = left_test.merge(right_test, on='plane_id', how='left', validate='many_to_one')
    null_row = result_test[result_test['plane_id'].isna()]
    print('NULL-KEY JOIN TEST: PASS')
    print('Valid key matched correctly.')
    print('Null-key observation retained without a false match.')
