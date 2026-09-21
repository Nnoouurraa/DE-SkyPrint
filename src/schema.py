
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
    import re
    import sys
    import shapely
    import subprocess
    import importlib.util
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_colwidth', 160)
    CWD = Path.cwd()
    PROJECT_ROOT = CWD.parent if CWD.name == 'notebooks' else CWD
    INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
    PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
    REFERENCE_DIR = PROJECT_ROOT / 'data' / 'reference'
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print('Project root:', PROJECT_ROOT.resolve())
    print('Interim dir:', INTERIM_DIR.resolve())
    import json
    from shapely.geometry import shape, Point
    boundary_path = REFERENCE_DIR / 'geoBoundaries-SAU-ADM0.geojson'
    with open(boundary_path, 'r', encoding='utf-8') as f:
        saudi_geojson = json.load(f)
    saudi_boundary = shape(saudi_geojson['features'][0]['geometry'])
    print('File exists:', boundary_path.exists())
    print('Geometry type:', saudi_boundary.geom_type)
    print('Geometry valid:', saudi_boundary.is_valid)
    paths = {'opensky': INTERIM_DIR / 'cleaned_opensky.csv', 'weather': INTERIM_DIR / 'cleaned_weather.csv', 'aircraft': INTERIM_DIR / 'cleaned_aircraft.csv'}
    missing_files = [str(p) for p in paths.values() if not p.exists()]
    if missing_files:
        raise FileNotFoundError('Missing Task 2 cleaned file(s):\n- ' + '\n- '.join(missing_files))
    df_sky = pd.read_csv(paths['opensky'], low_memory=False)
    df_wx = pd.read_csv(paths['weather'], low_memory=False)
    df_ac = pd.read_csv(paths['aircraft'], low_memory=False)
    print('OpenSky:', df_sky.shape)
    print('Weather:', df_wx.shape)
    print('Aircraft:', df_ac.shape)
    opensky_schema = pd.DataFrame([['plane_id', 'string', 'N', '6 lowercase hexadecimal characters', 'ICAO24 identifier; project join key'], ['flight_id', 'string', 'Y', 'text / null', 'Callsign may be unavailable'], ['origin_country', 'string', 'Y', 'text / null', 'Country inferred from ICAO24'], ['time_position', 'datetime UTC', 'Y', 'parseable timestamp / null', 'Source allows null; project position readiness requires it'], ['last_contact', 'datetime UTC', 'N', 'parseable timestamp', 'Last received transponder message'], ['longitude', 'float', 'Y', '-180 to 180 / null', 'WGS84 longitude'], ['latitude', 'float', 'Y', '-90 to 90 / null', 'WGS84 latitude'], ['baro_altitude', 'float', 'Y', 'numeric / null', 'Meters'], ['on_ground', 'boolean', 'N', 'True or False', 'Surface-position indicator'], ['velocity', 'float', 'Y', '>= 0 m/s / null', 'OpenSky velocity over ground'], ['true_track', 'float', 'Y', '0 <= value < 360 / null', 'Degrees clockwise from north'], ['vertical_rate', 'float', 'Y', 'numeric / null', 'm/s; positive climb, negative descent'], ['geo_altitude', 'float', 'Y', 'numeric / null', 'Meters'], ['source_type', 'integer', 'N', '{0,1,2,3}', 'ADS-B / ASTERIX / MLAT / FLARM'], ['category', 'integer', 'Y', '0 to 20 / null', 'OpenSky extended aircraft category']], columns=['Column', 'Data Type', 'Nullable', 'Allowed Values / Range', 'Project Note'])
    weather_schema = pd.DataFrame([['location_id', 'integer', 'N', '>= 0', 'Identity of each weather location'], ['observation_time', 'datetime UTC', 'N', 'parseable timestamp', 'Hourly weather observation/forecast time'], ['latitude', 'float', 'N', '-90 to 90', 'Returned Open-Meteo grid latitude'], ['longitude', 'float', 'N', '-180 to 180', 'Returned Open-Meteo grid longitude'], ['elevation_m', 'float', 'Y', 'numeric / null', 'Returned grid elevation'], ['timezone', 'string', 'Y', 'text / null', 'Timezone metadata'], ['temperature_*hPa', 'float', 'Y', 'numeric / null', 'Pressure-level temperature'], ['wind_speed_*hPa', 'float', 'Y', '>= 0 / null', 'Pressure-level wind speed'], ['wind_direction_*hPa', 'float', 'Y', '0 to 360 / null', 'Pressure-level wind direction'], ['geopotential_height_*hPa', 'float', 'Y', 'numeric / null', 'Height of pressure surface']], columns=['Column', 'Data Type', 'Nullable', 'Allowed Values / Range', 'Project Note'])
    aircraft_schema = pd.DataFrame([['plane_id', 'string', 'N', '6 hexadecimal characters', 'Join key to OpenSky'], ['registration', 'string', 'Y', 'text / null', 'Aircraft registration'], ['manufacturericao', 'string', 'Y', 'text / null', 'Manufacturer ICAO code'], ['manufacturername', 'string', 'Y', 'text / null', 'Manufacturer name'], ['model', 'string', 'Y', 'text / null', 'Aircraft model'], ['typecode', 'string', 'Y', 'normalized text / null', 'Aircraft type used for OpenAP readiness'], ['serialnumber', 'string', 'Y', 'text / null', 'Aircraft serial number'], ['icaoaircrafttype', 'string', 'Y', 'text / null', 'ICAO aircraft classification'], ['operator', 'string', 'Y', 'text / null', 'Aircraft operator'], ['operatoricao', 'string', 'Y', 'text / null', 'Operator ICAO code'], ['owner', 'string', 'Y', 'text / null', 'Aircraft owner'], ['registered', 'datetime', 'Y', 'parseable date / null', 'Registration date'], ['reguntil', 'datetime', 'Y', 'parseable date / null', 'Registration validity date'], ['built', 'datetime', 'Y', 'parseable date / null', 'Aircraft build date'], ['engines', 'string', 'Y', 'text / null', 'Engine metadata'], ['categoryDescription', 'string', 'Y', 'text / null', 'Aircraft category description'], ['typecode_status', 'string', 'N', '{matched, missing, conflict}', 'Task 2 duplicate-resolution status'], ['source_rows', 'integer', 'N', '>= 1', 'Number of source rows represented'], ['typecode_candidates', 'string', 'Y', 'pipe-separated candidate values / null', 'Preserves typecode evidence']], columns=['Column', 'Data Type', 'Nullable', 'Allowed Values / Range', 'Project Note'])
    print('OPEN SKY SCHEMA')
    display(opensky_schema)
    print('\nWEATHER SCHEMA')
    display(weather_schema)
    print('\nAIRCRAFT SCHEMA')
    display(aircraft_schema)

    def add_reason(reason_lists, mask, reason):
        mask = pd.Series(mask, index=reason_lists.index).fillna(False)
        for idx in reason_lists.index[mask]:
            reason_lists.at[idx].append(reason)

    def finalize_validation(df, reasons):
        out = df.copy()
        out['rejection_reason'] = reasons.apply(lambda x: ' | '.join(x) if x else pd.NA)
        out['schema_valid'] = out['rejection_reason'].isna()
        return out

    def numeric_conversion_failure(original, converted):
        return original.notna() & converted.isna()

    def datetime_conversion_failure(original, converted):
        return original.notna() & converted.isna()
    sky = df_sky.copy()
    reasons = pd.Series([[] for _ in range(len(sky))], index=sky.index, dtype='object')
    required_cols = ['plane_id', 'flight_id', 'origin_country', 'time_position', 'last_contact', 'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity', 'true_track', 'vertical_rate', 'geo_altitude', 'source_type', 'category']
    missing_cols = [c for c in required_cols if c not in sky.columns]
    if missing_cols:
        raise ValueError(f'OpenSky missing required columns: {missing_cols}')
    sky['plane_id'] = sky['plane_id'].astype('string').str.strip().str.lower()
    for col in ['time_position', 'last_contact']:
        original = sky[col].copy()
        converted = pd.to_datetime(original, utc=True, errors='coerce')
        add_reason(reasons, datetime_conversion_failure(original, converted), f'{col}: invalid datetime')
        sky[col] = converted
    numeric_cols = ['longitude', 'latitude', 'baro_altitude', 'velocity', 'true_track', 'vertical_rate', 'geo_altitude', 'source_type', 'category']
    for col in numeric_cols:
        original = sky[col].copy()
        converted = pd.to_numeric(original, errors='coerce')
        add_reason(reasons, numeric_conversion_failure(original, converted), f'{col}: not numeric')
        sky[col] = converted
    bool_map = {True: True, False: False, 1: True, 0: False, 'True': True, 'False': False, 'true': True, 'false': False, '1': True, '0': False}
    orig_ground = sky['on_ground'].copy()
    sky['on_ground'] = orig_ground.map(bool_map)
    add_reason(reasons, orig_ground.notna() & sky['on_ground'].isna(), 'on_ground: invalid boolean')
    add_reason(reasons, sky['plane_id'].isna(), 'plane_id: required')
    add_reason(reasons, sky['plane_id'].notna() & ~sky['plane_id'].str.fullmatch('[0-9a-f]{6}', na=False), 'plane_id: must be 6 hex characters')
    add_reason(reasons, sky['last_contact'].isna(), 'last_contact: required')
    add_reason(reasons, sky['on_ground'].isna(), 'on_ground: required')
    add_reason(reasons, sky['source_type'].isna(), 'source_type: required')
    add_reason(reasons, sky['latitude'].notna() & ~sky['latitude'].between(-90, 90), 'latitude: outside [-90, 90]')
    add_reason(reasons, sky['longitude'].notna() & ~sky['longitude'].between(-180, 180), 'longitude: outside [-180, 180]')
    add_reason(reasons, sky['velocity'].notna() & (sky['velocity'] < 0), 'velocity: must be >= 0')
    add_reason(reasons, sky['true_track'].notna() & ~((sky['true_track'] >= 0) & (sky['true_track'] < 360)), 'true_track: outside [0, 360)')
    add_reason(reasons, sky['source_type'].notna() & ~sky['source_type'].isin([0, 1, 2, 3]), 'source_type: allowed values are 0,1,2,3')
    add_reason(reasons, sky['category'].notna() & ~sky['category'].between(0, 20), 'category: outside [0, 20]')
    dup_key = sky.duplicated(['plane_id', 'time_position'], keep=False) & sky['time_position'].notna()
    add_reason(reasons, dup_key, 'duplicate natural key: plane_id + time_position')
    sky_v = finalize_validation(sky, reasons)
    sky_v['position_ready'] = sky_v['schema_valid'] & sky_v['time_position'].notna() & sky_v['latitude'].notna() & sky_v['longitude'].notna()
    sky_v['trajectory_point_ready'] = sky_v['position_ready'] & sky_v['plane_id'].notna()
    sky_v['kinematic_inputs_present'] = sky_v['position_ready'] & sky_v['velocity'].notna() & (sky_v['baro_altitude'].notna() | sky_v['geo_altitude'].notna())
    sky_v['inside_extraction_bbox'] = sky_v['latitude'].between(16.0, 33.0) & sky_v['longitude'].between(34.0, 56.0)

    def is_inside_saudi(row):
        if pd.isna(row['longitude']) or pd.isna(row['latitude']):
            return False
        point = Point(row['longitude'], row['latitude'])
        return saudi_boundary.covers(point)
    sky_v['inside_saudi_boundary'] = sky_v.apply(is_inside_saudi, axis=1)
    print('Schema validation:')
    print(sky_v['schema_valid'].value_counts(dropna=False))
    print('\nBusiness readiness:')
    print(sky_v[['position_ready', 'trajectory_point_ready', 'kinematic_inputs_present', 'inside_extraction_bbox']].sum())
    print('\nExact Saudi boundary:')
    print(sky_v['inside_saudi_boundary'].value_counts(dropna=False))
    print('\nCoordinate ranges:')
    print('Latitude :', sky_v['latitude'].min(), 'to', sky_v['latitude'].max())
    print('Longitude:', sky_v['longitude'].min(), 'to', sky_v['longitude'].max())
    wx = df_wx.copy()
    reasons = pd.Series([[] for _ in range(len(wx))], index=wx.index, dtype='object')
    base_required = ['location_id', 'observation_time', 'latitude', 'longitude', 'elevation_m', 'timezone']
    missing_cols = [c for c in base_required if c not in wx.columns]
    if missing_cols:
        raise ValueError(f'Weather missing required columns: {missing_cols}')
    pressure_levels = [850, 700, 500, 300, 250, 200]
    weather_variables = ['temperature', 'wind_speed', 'wind_direction', 'geopotential_height']
    expected_pressure_cols = [f'{variable}_{level}hPa' for level in pressure_levels for variable in weather_variables]
    missing_pressure_cols = [c for c in expected_pressure_cols if c not in wx.columns]
    if missing_pressure_cols:
        raise ValueError(f'Weather missing expected pressure-level columns: {missing_pressure_cols}')
    pressure_cols = expected_pressure_cols
    orig_time = wx['observation_time'].copy()
    wx['observation_time'] = pd.to_datetime(orig_time, utc=True, errors='coerce')
    add_reason(reasons, datetime_conversion_failure(orig_time, wx['observation_time']), 'observation_time: invalid datetime')
    numeric_cols = ['location_id', 'latitude', 'longitude', 'elevation_m'] + pressure_cols
    for col in numeric_cols:
        original = wx[col].copy()
        converted = pd.to_numeric(original, errors='coerce')
        add_reason(reasons, numeric_conversion_failure(original, converted), f'{col}: not numeric')
        wx[col] = converted
    add_reason(reasons, wx['location_id'].isna(), 'location_id: required')
    add_reason(reasons, wx['observation_time'].isna(), 'observation_time: required')
    add_reason(reasons, wx['latitude'].isna(), 'latitude: required')
    add_reason(reasons, wx['longitude'].isna(), 'longitude: required')
    add_reason(reasons, wx['location_id'].notna() & (wx['location_id'] < 0), 'location_id: must be >= 0')
    add_reason(reasons, wx['latitude'].notna() & ~wx['latitude'].between(-90, 90), 'latitude: outside [-90, 90]')
    add_reason(reasons, wx['longitude'].notna() & ~wx['longitude'].between(-180, 180), 'longitude: outside [-180, 180]')
    wind_speed_cols = [f'wind_speed_{level}hPa' for level in pressure_levels]
    for col in wind_speed_cols:
        add_reason(reasons, wx[col].notna() & (wx[col] < 0), f'{col}: must be >= 0')
    wind_dir_cols = [f'wind_direction_{level}hPa' for level in pressure_levels]
    for col in wind_dir_cols:
        add_reason(reasons, wx[col].notna() & ~wx[col].between(0, 360), f'{col}: outside [0, 360]')
    dup_key = wx.duplicated(['location_id', 'observation_time'], keep=False) & wx['location_id'].notna() & wx['observation_time'].notna()
    add_reason(reasons, dup_key, 'duplicate natural key: location_id + observation_time')
    wx_v = finalize_validation(wx, reasons)
    temp_cols = [f'temperature_{level}hPa' for level in pressure_levels]
    height_cols = [f'geopotential_height_{level}hPa' for level in pressure_levels]
    wx_v['pressure_weather_available'] = wx_v[temp_cols].notna().any(axis=1) & wx_v[wind_speed_cols].notna().any(axis=1) & wx_v[wind_dir_cols].notna().any(axis=1) & wx_v[height_cols].notna().any(axis=1)
    wx_v['weather_ready'] = wx_v['schema_valid'] & wx_v['pressure_weather_available']
    print('Schema validation:')
    print(wx_v['schema_valid'].value_counts(dropna=False))
    print('\nWeather-ready rows:', int(wx_v['weather_ready'].sum()), '/', len(wx_v))
    print('Unique location IDs:', wx_v['location_id'].nunique(dropna=True))
    ac = df_ac.copy()
    reasons = pd.Series([[] for _ in range(len(ac))], index=ac.index, dtype='object')
    required_cols = ['plane_id', 'typecode', 'typecode_status', 'source_rows', 'typecode_candidates']
    missing_cols = [c for c in required_cols if c not in ac.columns]
    if missing_cols:
        raise ValueError(f'Aircraft missing required columns: {missing_cols}')
    ac['plane_id'] = ac['plane_id'].astype('string').str.strip().str.lower()
    ac['typecode'] = ac['typecode'].astype('string').str.strip().str.upper().replace({'': pd.NA, 'NAN': pd.NA, 'NONE': pd.NA, 'NULL': pd.NA})
    ac['typecode_status'] = ac['typecode_status'].astype('string').str.strip().str.lower()
    orig_source_rows = ac['source_rows'].copy()
    ac['source_rows'] = pd.to_numeric(orig_source_rows, errors='coerce')
    add_reason(reasons, numeric_conversion_failure(orig_source_rows, ac['source_rows']), 'source_rows: not numeric')
    add_reason(reasons, ac['plane_id'].isna(), 'plane_id: required')
    add_reason(reasons, ac['plane_id'].notna() & ~ac['plane_id'].str.fullmatch('[0-9a-f]{6}', na=False), 'plane_id: must be 6 hex characters')
    add_reason(reasons, ac['typecode_status'].isna(), 'typecode_status: required')
    add_reason(reasons, ac['typecode_status'].notna() & ~ac['typecode_status'].isin(['matched', 'missing', 'conflict']), 'typecode_status: invalid value')
    add_reason(reasons, ac['source_rows'].isna(), 'source_rows: required')
    add_reason(reasons, ac['source_rows'].notna() & (ac['source_rows'] < 1), 'source_rows: must be >= 1')
    add_reason(reasons, ac['plane_id'].notna() & ac['plane_id'].duplicated(keep=False), 'duplicate plane_id after cleaning')
    add_reason(reasons, (ac['typecode_status'] == 'matched') & ac['typecode'].isna(), 'matched status requires typecode')
    add_reason(reasons, (ac['typecode_status'] == 'missing') & ac['typecode'].notna(), 'missing status should have null typecode')
    add_reason(reasons, (ac['typecode_status'] == 'conflict') & ac['typecode'].notna(), 'conflict status should keep typecode null')
    add_reason(reasons, (ac['typecode_status'] == 'conflict') & ac['typecode_candidates'].isna(), 'conflict status requires preserved candidates')
    ac_v = finalize_validation(ac, reasons)
    print('Schema validation:')
    print(ac_v['schema_valid'].value_counts(dropna=False))
    if importlib.util.find_spec('openap') is None:
        print('OpenAP not found. Installing...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openap'])
    from openap import prop, FuelFlow
    supported_direct = {str(x).upper() for x in prop.available_aircraft(use_synonym=False)}
    supported_with_synonyms = {str(x).upper() for x in prop.available_aircraft(use_synonym=True)}
    print('OpenAP direct aircraft types:', len(supported_direct))
    print('OpenAP types including synonyms:', len(supported_with_synonyms))
    print('Sample:', sorted(supported_with_synonyms)[:20])
    fuel_support_cache = {}

    def fuel_model_supported(code):
        if code is None or pd.isna(code):
            return False
        code = str(code).upper().strip()
        if code in fuel_support_cache:
            return fuel_support_cache[code]
        try:
            FuelFlow(ac=code, use_synonym=True)
            fuel_support_cache[code] = True
        except Exception:
            fuel_support_cache[code] = False
        return fuel_support_cache[code]

    def supported_tokens(text):
        if text is None or pd.isna(text):
            return []
        text = str(text).upper().strip()
        found = []
        for code in supported_with_synonyms:
            if re.search(f'(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])', text):
                found.append(code)
        return sorted(set(found))

    def resolve_single_typecode(value):
        if value is None or pd.isna(value):
            return {'openap_typecode': pd.NA, 'resolution_method': 'missing', 'openap_supported': False, 'resolution_note': 'No usable typecode'}
        value = str(value).upper().strip()
        if value in supported_with_synonyms and fuel_model_supported(value):
            return {'openap_typecode': value, 'resolution_method': 'exact_openap', 'openap_supported': True, 'resolution_note': 'Exact OpenAP-supported code/synonym'}
        tokens = [t for t in supported_tokens(value) if fuel_model_supported(t)]
        if len(tokens) == 1:
            return {'openap_typecode': tokens[0], 'resolution_method': 'supported_token', 'openap_supported': True, 'resolution_note': f'Extracted one unambiguous OpenAP token from: {value}'}
        if len(tokens) > 1:
            return {'openap_typecode': pd.NA, 'resolution_method': 'ambiguous_tokens', 'openap_supported': False, 'resolution_note': 'Multiple OpenAP-supported tokens found: ' + '|'.join(tokens)}
        return {'openap_typecode': pd.NA, 'resolution_method': 'unsupported', 'openap_supported': False, 'resolution_note': f'No supported OpenAP aircraft code found in: {value}'}

    def resolve_aircraft_row(row):
        status = row['typecode_status']
        if status == 'conflict':
            raw = row.get('typecode_candidates', pd.NA)
            candidates = [] if pd.isna(raw) else [x.strip().upper() for x in str(raw).split('|') if x.strip()]
            resolved = []
            details = []
            for candidate in candidates:
                r = resolve_single_typecode(candidate)
                details.append(f"{candidate}:{r['resolution_method']}")
                if r['openap_supported']:
                    resolved.append(str(r['openap_typecode']).upper())
            unique_resolved = sorted(set(resolved))
            if len(unique_resolved) == 1:
                return pd.Series({'openap_typecode': unique_resolved[0], 'openap_resolution': 'conflict_resolved_single_openap', 'openap_supported': True, 'openap_resolution_note': 'Conflict candidates resolve to one OpenAP code; ' + '; '.join(details)})
            if len(unique_resolved) > 1:
                return pd.Series({'openap_typecode': pd.NA, 'openap_resolution': 'conflict_ambiguous_openap', 'openap_supported': False, 'openap_resolution_note': 'Multiple different OpenAP codes remain: ' + '|'.join(unique_resolved)})
            return pd.Series({'openap_typecode': pd.NA, 'openap_resolution': 'conflict_no_openap_match', 'openap_supported': False, 'openap_resolution_note': 'No conflict candidate is supported by OpenAP'})
        r = resolve_single_typecode(row['typecode'])
        return pd.Series({'openap_typecode': r['openap_typecode'], 'openap_resolution': r['resolution_method'], 'openap_supported': r['openap_supported'], 'openap_resolution_note': r['resolution_note']})
    resolution_keys = ac_v[['typecode', 'typecode_status', 'typecode_candidates']].drop_duplicates().copy()
    resolution_result = resolution_keys.apply(resolve_aircraft_row, axis=1)
    resolution_map = pd.concat([resolution_keys.reset_index(drop=True), resolution_result.reset_index(drop=True)], axis=1)
    ac_v = ac_v.merge(resolution_map, on=['typecode', 'typecode_status', 'typecode_candidates'], how='left', validate='many_to_one')
    ac_v['metadata_ready'] = ac_v['schema_valid']
    ac_v['fuel_model_ready'] = ac_v['schema_valid'] & ac_v['openap_supported'].fillna(False)
    print('\nOpenAP resolution:')
    print(ac_v['openap_resolution'].value_counts(dropna=False))
    print('\nFuel-model ready:', int(ac_v['fuel_model_ready'].sum()), '/', len(ac_v))
    decision_issues = []
    bad_supported_null = ac_v['openap_supported'].fillna(False) & ac_v['openap_typecode'].isna()
    if bad_supported_null.any():
        decision_issues.append(f'{bad_supported_null.sum()} rows say OpenAP-supported but have no openap_typecode')
    bad_ready = ac_v['fuel_model_ready'] & ~ac_v['schema_valid']
    if bad_ready.any():
        decision_issues.append(f'{bad_ready.sum()} fuel-ready rows are schema-invalid')
    chosen_codes = sorted(ac_v.loc[ac_v['openap_supported'].fillna(False), 'openap_typecode'].dropna().unique())
    failed_codes = [c for c in chosen_codes if not fuel_model_supported(c)]
    if failed_codes:
        decision_issues.append('Chosen codes that failed FuelFlow initialization: ' + ', '.join(failed_codes))
    if decision_issues:
        print('OPENAP DECISION CHECK: FAILED')
        for issue in decision_issues:
            print('-', issue)
    else:
        print('OPENAP DECISION CHECK: PASS')
        print('All selected OpenAP codes successfully initialize FuelFlow.')
        print('Selected unique OpenAP codes:', len(chosen_codes))
    ac_lookup_cols = ['plane_id', 'schema_valid', 'typecode', 'typecode_status', 'typecode_candidates', 'openap_typecode', 'openap_resolution', 'openap_supported', 'fuel_model_ready']
    ac_lookup = ac_v[ac_lookup_cols].rename(columns={'schema_valid': 'aircraft_schema_valid'})
    project_ready = sky_v.merge(ac_lookup, on='plane_id', how='left', validate='many_to_one')
    project_ready['aircraft_metadata_matched'] = project_ready['aircraft_schema_valid'].fillna(False)
    project_ready['openap_ready'] = project_ready['openap_supported'].fillna(False)
    project_ready['ready_for_task4_integration'] = project_ready['schema_valid'] & project_ready['position_ready'] & project_ready['kinematic_inputs_present'] & project_ready['aircraft_metadata_matched'] & project_ready['openap_ready']
    print('Snapshot rows:', len(project_ready))
    print('Aircraft metadata matched:', int(project_ready['aircraft_metadata_matched'].sum()))
    print('OpenAP ready:', int(project_ready['openap_ready'].sum()))
    print('Ready for Task 4 integration:', int(project_ready['ready_for_task4_integration'].sum()))
    sky_validated = sky_v[sky_v['schema_valid']].copy()
    sky_rejected = sky_v[~sky_v['schema_valid']].copy()
    wx_validated = wx_v[wx_v['schema_valid']].copy()
    wx_rejected = wx_v[~wx_v['schema_valid']].copy()
    ac_validated = ac_v[ac_v['schema_valid']].copy()
    ac_rejected = ac_v[~ac_v['schema_valid']].copy()
    outputs = {INTERIM_DIR / 'validated.csv': sky_validated, INTERIM_DIR / 'rejected.csv': sky_rejected, INTERIM_DIR / 'validated_opensky.csv': sky_validated, INTERIM_DIR / 'rejected_opensky.csv': sky_rejected, INTERIM_DIR / 'validated_weather.csv': wx_validated, INTERIM_DIR / 'rejected_weather.csv': wx_rejected, INTERIM_DIR / 'validated_aircraft.csv': ac_validated, INTERIM_DIR / 'rejected_aircraft.csv': ac_rejected, INTERIM_DIR / 'skyprint_readiness.csv': project_ready}
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
        print(f'{path.name:<30} rows={len(frame):>8,}')
    summary = pd.DataFrame([{'dataset': 'OpenSky', 'input_rows': len(sky_v), 'validated_rows': len(sky_validated), 'rejected_rows': len(sky_rejected), 'reconciles': len(sky_v) == len(sky_validated) + len(sky_rejected), 'business_ready_rows': int(project_ready['ready_for_task4_integration'].sum())}, {'dataset': 'Weather', 'input_rows': len(wx_v), 'validated_rows': len(wx_validated), 'rejected_rows': len(wx_rejected), 'reconciles': len(wx_v) == len(wx_validated) + len(wx_rejected), 'business_ready_rows': int(wx_v['weather_ready'].sum())}, {'dataset': 'Aircraft', 'input_rows': len(ac_v), 'validated_rows': len(ac_validated), 'rejected_rows': len(ac_rejected), 'reconciles': len(ac_v) == len(ac_validated) + len(ac_rejected), 'business_ready_rows': int(ac_v['fuel_model_ready'].sum())}])
    display(summary)
    summary.to_csv(INTERIM_DIR / 'validation_summary.csv', index=False)
    opensky_schema.to_csv(INTERIM_DIR / 'schema_opensky.csv', index=False)
    weather_schema.to_csv(INTERIM_DIR / 'schema_weather.csv', index=False)
    aircraft_schema.to_csv(INTERIM_DIR / 'schema_aircraft.csv', index=False)
    all_rejected_have_reason = all([sky_rejected['rejection_reason'].notna().all(), wx_rejected['rejection_reason'].notna().all(), ac_rejected['rejection_reason'].notna().all()])
    checks = {'All datasets reconcile': bool(summary['reconciles'].all()), 'Every rejected row has a reason': bool(all_rejected_have_reason), 'OpenSky natural key duplicates remaining': int(sky_validated.duplicated(['plane_id', 'time_position']).sum()), 'Weather natural key duplicates remaining': int(wx_validated.duplicated(['location_id', 'observation_time']).sum()), 'Aircraft duplicate plane IDs remaining': int(ac_validated['plane_id'].duplicated().sum()), 'OpenAP resolver internal issues': len(decision_issues)}
    print('\n=== FINAL VALIDATION CHECK ===')
    for k, v in checks.items():
        print(f'{k}: {v}')
    hard_pass = checks['All datasets reconcile'] and checks['Every rejected row has a reason'] and (checks['OpenSky natural key duplicates remaining'] == 0) and (checks['Weather natural key duplicates remaining'] == 0) and (checks['Aircraft duplicate plane IDs remaining'] == 0) and (checks['OpenAP resolver internal issues'] == 0)
    print('\nRESULT:', 'PASS' if hard_pass else 'REVIEW REQUIRED')
    print('OPEN SKY REJECTION REASONS')
    display(sky_rejected['rejection_reason'].value_counts(dropna=False).to_frame('rows'))
    print('\nWEATHER REJECTION REASONS')
    display(wx_rejected['rejection_reason'].value_counts(dropna=False).to_frame('rows'))
    print('\nAIRCRAFT REJECTION REASONS')
    display(ac_rejected['rejection_reason'].value_counts(dropna=False).to_frame('rows'))
    print('\nOPENAP RESOLUTION STATUS')
    display(ac_v['openap_resolution'].value_counts(dropna=False).to_frame('rows'))
    print('\nCURRENT SNAPSHOT — NOT READY FOR TASK 4')
    cols = ['plane_id', 'flight_id', 'position_ready', 'kinematic_inputs_present', 'aircraft_metadata_matched', 'typecode', 'typecode_status', 'openap_typecode', 'openap_resolution', 'openap_ready', 'ready_for_task4_integration']
    display(project_ready.loc[~project_ready['ready_for_task4_integration'], cols].head(5))
    print('=== CURRENT AIRCRAFT READINESS ===')
    print('Total OpenSky aircraft:', len(project_ready))
    print('Matched with Aircraft Database:', project_ready['aircraft_metadata_matched'].sum())
    print('Matched + supported by OpenAP:', project_ready['openap_ready'].sum())
    print('Fully ready for Task 4:', project_ready['ready_for_task4_integration'].sum())
