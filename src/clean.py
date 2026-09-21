
from __future__ import annotations

def _display(obj):
    """Lightweight replacement for Jupyter display() during CLI runs."""
    try:
        print(obj.to_string())
    except AttributeError:
        print(obj)

def run():
    display = _display
    import pandas as pd
    import numpy as np
    import json
    from pathlib import Path
    CWD = Path.cwd()
    PROJECT_ROOT = CWD.parent if CWD.name == 'notebooks' else CWD
    RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
    INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.max_columns', None)

    def profile(df):
        rows = []
        for c in df.columns:
            s = df[c]
            sample = s.dropna().iloc[0] if s.notna().any() else None
            rows.append({'column': c, 'dtype': str(s.dtype), 'null_count': int(s.isna().sum()), 'percent_null': round(100 * s.isna().sum() / len(s), 2) if len(s) else 0, 'unique_count': int(s.nunique(dropna=True)), 'sample_value': sample})
        return pd.DataFrame(rows)

    def latest_timestamped_file(prefix):
        files = list(RAW_DIR.glob(f'{prefix}_*.json'))
        if not files:
            fallback = RAW_DIR / f'{prefix}.json'
            if fallback.exists():
                return fallback
            raise FileNotFoundError(f'No {prefix} JSON file found in {RAW_DIR}')
        return max(files, key=lambda p: p.stat().st_mtime)
    opensky_path = latest_timestamped_file('opensky')
    with open(opensky_path, encoding='utf-8') as f:
        opensky_raw = json.load(f)
    print('loaded file:', opensky_path.name)
    print('snapshot unix time:', opensky_raw.get('time'))
    print('number of state vectors:', len(opensky_raw.get('states', [])))
    if opensky_raw.get('states'):
        print('fields in first state vector:', len(opensky_raw['states'][0]))
        display(opensky_raw['states'][0])
    OPENSKY_COLUMNS = ['plane_id', 'flight_id', 'origin_country', 'time_position', 'last_contact', 'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity', 'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'source_type', 'category']
    state_lengths = pd.Series([len(row) for row in opensky_raw.get('states', [])]).value_counts().sort_index()
    print('state-vector lengths:')
    print(state_lengths)
    if len(state_lengths) and (not (len(state_lengths) == 1 and state_lengths.index[0] == 18)):
        raise ValueError('Unexpected OpenSky state-vector length. Expected 18 fields because extended=1 is used.')
    df_sky = pd.DataFrame(opensky_raw.get('states', []), columns=OPENSKY_COLUMNS)
    print('shape:', df_sky.shape)
    df_sky.head()
    profile_sky = profile(df_sky)
    profile_sky
    print('rows:', len(df_sky), ' columns:', df_sky.shape[1])
    print('exact duplicate rows:', df_sky.duplicated().sum())
    print('duplicate (plane_id, time_position) pairs:', df_sky.duplicated(subset=['plane_id', 'time_position']).sum())
    flight_text = df_sky['flight_id'].astype('string')
    print('empty-string flight_ids:', (flight_text.str.strip() == '').sum())
    print('missing positions:', df_sky[['latitude', 'longitude']].isna().any(axis=1).sum())
    print('category value counts:')
    print(df_sky['category'].value_counts(dropna=False).sort_index())
    df_sky_clean = df_sky.copy()
    df_sky_clean = df_sky_clean.drop(columns=['sensors', 'squawk', 'spi'])
    df_sky_clean['plane_id'] = df_sky_clean['plane_id'].astype('string').str.strip().str.lower()
    df_sky_clean['flight_id'] = df_sky_clean['flight_id'].astype('string').str.strip().replace('', pd.NA)
    df_sky_clean['origin_country'] = df_sky_clean['origin_country'].astype('string').str.strip().replace('', pd.NA)
    for col in ['time_position', 'last_contact']:
        df_sky_clean[col] = pd.to_datetime(df_sky_clean[col], unit='s', utc=True, errors='coerce')
    df_sky_clean['on_ground'] = df_sky_clean['on_ground'].astype('boolean')
    df_sky_clean['category'] = pd.to_numeric(df_sky_clean['category'], errors='coerce').astype('Int64')
    df_sky_clean['source_type'] = pd.to_numeric(df_sky_clean['source_type'], errors='coerce').astype('Int64')
    before = len(df_sky_clean)
    df_sky_clean = df_sky_clean.drop_duplicates(subset=['plane_id', 'time_position'])
    print(f'dropped {before - len(df_sky_clean)} duplicate OpenSky observations')
    df_sky_clean.dtypes
    df_sky_clean.head()
    sky_output = INTERIM_DIR / 'cleaned_opensky.csv'
    df_sky_clean.to_csv(sky_output, index=False)
    print('saved:', sky_output.resolve())
    meteo_path = latest_timestamped_file('openmeteo')
    with open(meteo_path, encoding='utf-8') as f:
        meteo_raw = json.load(f)
    print('loaded file:', meteo_path.name)
    print('raw response type:', type(meteo_raw).__name__)
    meteo_locations = meteo_raw if isinstance(meteo_raw, list) else [meteo_raw]
    print('weather locations returned:', len(meteo_locations))
    weather_frames = []
    for location in meteo_locations:
        hourly = location.get('hourly', {})
        if not hourly or 'time' not in hourly:
            continue
        frame = pd.DataFrame(hourly)
        frame['latitude'] = location.get('latitude')
        frame['longitude'] = location.get('longitude')
        frame['elevation_m'] = location.get('elevation')
        frame['location_id'] = location.get('location_id', 0)
        frame['timezone'] = location.get('timezone')
        weather_frames.append(frame)
    if not weather_frames:
        raise ValueError('No hourly weather data found in the Open-Meteo raw response.')
    df_wx = pd.concat(weather_frames, ignore_index=True)
    print('shape:', df_wx.shape)
    print('unique weather locations:', df_wx[['latitude', 'longitude']].drop_duplicates().shape[0])
    df_wx.head()
    print('Forecast objects returned:', len(meteo_locations))
    coords = [(loc.get('latitude'), loc.get('longitude')) for loc in meteo_locations]
    print('Unique returned coordinates:', len(set(coords)))
    from collections import Counter
    duplicates = [coord for coord, count in Counter(coords).items() if count > 1]
    print('Repeated coordinates:', duplicates)
    profile_wx = profile(df_wx)
    profile_wx
    print('rows:', len(df_wx))
    print('exact duplicate rows:', df_wx.duplicated().sum())
    print('unique locations:', df_wx[['latitude', 'longitude']].drop_duplicates().shape[0])
    print('duplicate (location, time) rows:', df_wx.duplicated(subset=['latitude', 'longitude', 'time']).sum())
    print('missing timestamps:', df_wx['time'].isna().sum())
    weather_dup_mask = df_wx.duplicated(subset=['latitude', 'longitude', 'time'], keep=False)
    weather_duplicates = df_wx[weather_dup_mask].sort_values(['latitude', 'longitude', 'time'])
    print('Rows involved in duplicates:', len(weather_duplicates))
    print('Duplicated location/time groups:', weather_duplicates.groupby(['latitude', 'longitude', 'time']).ngroups)
    weather_duplicates
    df_wx_clean = df_wx.copy()
    df_wx_clean['time'] = pd.to_datetime(df_wx_clean['time'], utc=True, errors='coerce')
    df_wx_clean = df_wx_clean.rename(columns={'time': 'observation_time'})
    non_numeric_weather = {'observation_time', 'timezone'}
    for col in df_wx_clean.columns:
        if col not in non_numeric_weather:
            df_wx_clean[col] = pd.to_numeric(df_wx_clean[col], errors='coerce')
    exact_duplicates = df_wx_clean.duplicated().sum()
    print('Exact duplicate weather rows:', exact_duplicates)
    if exact_duplicates > 0:
        df_wx_clean = df_wx_clean.drop_duplicates()
    print('Weather rows after cleaning:', len(df_wx_clean))
    df_wx_clean.head()
    weather_output = INTERIM_DIR / 'cleaned_weather.csv'
    df_wx_clean.to_csv(weather_output, index=False)
    print('saved:', weather_output.resolve())
    df_ac = pd.read_csv(RAW_DIR / 'aircraftDatabase.csv', dtype='string')
    df_ac = df_ac.rename(columns={'icao24': 'plane_id'})
    print('shape:', df_ac.shape)
    df_ac.head(3)
    profile_ac = profile(df_ac)
    profile_ac
    null_pct = (df_ac.isna().sum() / len(df_ac) * 100).round(2).sort_values(ascending=False)
    print('rows with plane_id null:', df_ac['plane_id'].isna().sum())
    print('exact duplicate rows:', df_ac.duplicated().sum())
    print('duplicate plane_id rows (before normalization):', df_ac[df_ac['plane_id'].notna()].duplicated(subset=['plane_id']).sum())
    print('typecode nulls:', df_ac['typecode'].isna().sum())
    print()
    print('columns with >= 90% nulls:')
    print(null_pct[null_pct >= 90])
    df_ac_clean = df_ac.copy()
    df_ac_clean['plane_id'] = df_ac_clean['plane_id'].astype('string').str.strip().str.lower().replace({'': pd.NA, 'nan': pd.NA, 'none': pd.NA, 'null': pd.NA})
    df_ac_clean = df_ac_clean.dropna(subset=['plane_id'])
    df_ac_clean['typecode'] = df_ac_clean['typecode'].astype('string').str.strip().str.upper().replace({'': pd.NA, 'NAN': pd.NA, 'NONE': pd.NA, 'NULL': pd.NA})
    before = len(df_ac_clean)
    df_ac_clean = df_ac_clean.drop_duplicates()
    print(f'dropped {before - len(df_ac_clean)} exact duplicate aircraft rows after normalization')
    cols_to_drop = ['modes', 'adsb', 'acars', 'status', 'seatconfiguration', 'firstflightdate', 'testreg', 'notes', 'linenumber', 'operatoriata', 'operatorcallsign']
    cols_to_drop = [c for c in cols_to_drop if c in df_ac_clean.columns]
    df_ac_clean = df_ac_clean.drop(columns=cols_to_drop)
    for col in ['built', 'registered', 'reguntil']:
        if col in df_ac_clean.columns:
            df_ac_clean[col] = pd.to_datetime(df_ac_clean[col], errors='coerce')
    typecodes_per_plane = df_ac_clean.groupby('plane_id')['typecode'].agg(lambda s: sorted(set(s.dropna())))
    duplicate_plane_ids = df_ac_clean.loc[df_ac_clean.duplicated('plane_id', keep=False), 'plane_id'].unique()
    print('duplicate plane_ids after normalization:', len(duplicate_plane_ids))
    duplicate_typecode_summary = pd.DataFrame({'plane_id': typecodes_per_plane.index, 'unique_non_null_typecodes': typecodes_per_plane.values})
    duplicate_typecode_summary['typecode_count'] = duplicate_typecode_summary['unique_non_null_typecodes'].str.len()
    duplicate_typecode_summary = duplicate_typecode_summary[duplicate_typecode_summary['plane_id'].isin(duplicate_plane_ids)]
    duplicate_typecode_summary.head(20)
    duplicate_mask = df_ac_clean.duplicated(subset=['plane_id'], keep=False)
    df_unique = df_ac_clean[~duplicate_mask].copy()
    df_duplicates = df_ac_clean[duplicate_mask].copy()
    df_unique['typecode_status'] = np.where(df_unique['typecode'].notna(), 'matched', 'missing')
    df_unique['source_rows'] = 1
    df_unique['typecode_candidates'] = df_unique['typecode']
    resolved_duplicates = []
    for plane_id, group in df_duplicates.groupby('plane_id'):
        valid_typecodes = sorted(set(group['typecode'].dropna()))
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
        result['typecode_candidates'] = '|'.join(valid_typecodes) if valid_typecodes else pd.NA
        resolved_duplicates.append(result)
    df_resolved_duplicates = pd.DataFrame(resolved_duplicates)
    df_ac_clean = pd.concat([df_unique, df_resolved_duplicates], ignore_index=True)
    print('final aircraft rows:', len(df_ac_clean))
    print('unique plane_ids:', df_ac_clean['plane_id'].nunique())
    print('\ntypecode status:')
    print(df_ac_clean['typecode_status'].value_counts())
    print('\nremaining duplicate plane_ids:', df_ac_clean.duplicated(subset=['plane_id']).sum())
    conflicts = df_ac_clean[df_ac_clean['typecode_status'] == 'conflict'][['plane_id', 'typecode', 'typecode_candidates', 'source_rows']]
    print('real typecode conflicts:', len(conflicts))
    conflicts.head(20)
    df_ac_clean.head()
    print('Columns after cleaning:')
    print(df_ac_clean.columns.tolist())
    print('\nDate column types:')
    for col in ['built', 'registered', 'reguntil']:
        print(col, ':', df_ac_clean[col].dtype)
    aircraft_output = INTERIM_DIR / 'cleaned_aircraft.csv'
    df_ac_clean.to_csv(aircraft_output, index=False)
    print('saved:', aircraft_output.resolve())
    summary = pd.DataFrame([{'dataset': 'OpenSky states', 'raw_rows': len(df_sky), 'cleaned_rows': len(df_sky_clean), 'output': 'data/interim/cleaned_opensky.csv'}, {'dataset': 'Open-Meteo hourly by location', 'raw_rows': len(df_wx), 'cleaned_rows': len(df_wx_clean), 'output': 'data/interim/cleaned_weather.csv'}, {'dataset': 'Aircraft type lookup', 'raw_rows': len(df_ac), 'cleaned_rows': len(df_ac_clean), 'output': 'data/interim/cleaned_aircraft.csv'}])
    summary
    print(df_ac_clean.columns.tolist())
    print('=== FINAL CLEANING CHECK ===')
    print('----------------------------')
    print('OpenSky rows:', len(df_sky_clean))
    print('OpenSky duplicate keys:', df_sky_clean.duplicated(['plane_id', 'time_position']).sum())
    print('\nWeather rows:', len(df_wx_clean))
    print('Weather location IDs:', df_wx_clean['location_id'].nunique())
    print('Weather duplicate keys:', df_wx_clean.duplicated(['location_id', 'observation_time']).sum())
    print('\nAircraft rows:', len(df_ac_clean))
    print('Aircraft duplicate IDs:', df_ac_clean['plane_id'].duplicated().sum())
    print('Aircraft typecode conflicts:', (df_ac_clean['typecode_status'] == 'conflict').sum())
    print('\nTypecode status:')
    print(df_ac_clean['typecode_status'].value_counts(dropna=False))
