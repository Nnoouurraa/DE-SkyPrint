
from __future__ import annotations

def _display(obj):
    """Lightweight replacement for Jupyter display() during CLI runs."""
    try:
        print(obj.to_string())
    except AttributeError:
        print(obj)

def run():
    display = _display
    print('kernel works')
    import requests
    import json
    import pandas as pd
    from pathlib import Path
    from datetime import datetime, timezone
    from time import perf_counter
    CWD = Path.cwd()
    PROJECT_ROOT = CWD.parent if CWD.name == 'notebooks' else CWD
    raw_path = PROJECT_ROOT / 'data' / 'raw'
    logs_path = PROJECT_ROOT / 'logs'
    raw_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)
    extraction_time = datetime.now(timezone.utc)
    timestamp = extraction_time.strftime('%Y-%m-%d_%H-%M-%S')
    print('Raw data folder is ready.')
    print('Logs folder is ready.')
    print('Extraction time (UTC):', extraction_time)
    log_file = logs_path / 'extraction_log.csv'

    def write_extraction_log(source, status, rows, file_size_mb, duration_seconds, status_code=None, error_message=None):
        log_row = {'extraction_time_utc': extraction_time.isoformat(), 'source': source, 'status': status, 'rows_extracted': rows, 'file_size_mb': file_size_mb, 'duration_seconds': round(duration_seconds, 2), 'status_code': status_code, 'error_message': error_message}
        log_df = pd.DataFrame([log_row])
        if log_file.exists():
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)
    opensky_url = 'https://opensky-network.org/api/states/all'
    params = {'lamin': 16.0, 'lomin': 34.0, 'lamax': 33.0, 'lomax': 56.0, 'extended': 1}
    start_time = perf_counter()
    states = []
    opensky_data = None
    try:
        response = requests.get(opensky_url, params=params, timeout=30)
        response.raise_for_status()
        opensky_data = response.json()
        duration = perf_counter() - start_time
        states = opensky_data.get('states') or []
        aircraft_count = len(states)
        opensky_file = raw_path / f'opensky_{timestamp}.json'
        with open(opensky_file, 'w', encoding='utf-8') as f:
            json.dump(opensky_data, f)
        file_size_mb = opensky_file.stat().st_size / (1024 * 1024)
        print('OpenSky Status Code:', response.status_code)
        print('Number of aircraft:', aircraft_count)
        print('File size:', round(file_size_mb, 3), 'MB')
        print('Request duration:', round(duration, 2), 'seconds')
        print('OpenSky data saved to:', opensky_file)
        write_extraction_log(source='OpenSky', status='SUCCESS', rows=aircraft_count, file_size_mb=round(file_size_mb, 3), duration_seconds=duration, status_code=response.status_code)
    except requests.exceptions.Timeout:
        duration = perf_counter() - start_time
        print('OpenSky request timed out.')
        write_extraction_log(source='OpenSky', status='FAILED', rows=0, file_size_mb=0, duration_seconds=duration, error_message='Request timed out')
    except requests.exceptions.HTTPError as e:
        duration = perf_counter() - start_time
        status_code = response.status_code if 'response' in locals() else None
        print('OpenSky HTTP error:', e)
        write_extraction_log(source='OpenSky', status='FAILED', rows=0, file_size_mb=0, duration_seconds=duration, status_code=status_code, error_message=str(e))
    except requests.exceptions.RequestException as e:
        duration = perf_counter() - start_time
        print('OpenSky request failed:', e)
        write_extraction_log(source='OpenSky', status='FAILED', rows=0, file_size_mb=0, duration_seconds=duration, error_message=str(e))
    opensky_columns = ['icao24', 'callsign', 'origin_country', 'time_position', 'last_contact', 'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity', 'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source', 'category']
    if states:
        aircraft_df = pd.DataFrame(states, columns=opensky_columns)
        aircraft_locations = aircraft_df[['icao24', 'latitude', 'longitude', 'geo_altitude', 'time_position']].dropna(subset=['latitude', 'longitude']).copy()
        print('Aircraft with usable locations:', len(aircraft_locations))
        display(aircraft_locations.head())
    else:
        aircraft_locations = pd.DataFrame()
        print('No aircraft locations available.')
    if not aircraft_locations.empty:
        aircraft_locations['weather_latitude'] = aircraft_locations['latitude'].round(1)
        aircraft_locations['weather_longitude'] = aircraft_locations['longitude'].round(1)
        weather_locations = aircraft_locations[['weather_latitude', 'weather_longitude']].drop_duplicates().reset_index(drop=True)
        print('Aircraft locations:', len(aircraft_locations))
        print('Unique weather locations:', len(weather_locations))
    else:
        weather_locations = pd.DataFrame()
        print('No weather locations available.')
    weather_url = 'https://api.open-meteo.com/v1/forecast'
    weather_variables = ['temperature_850hPa', 'wind_speed_850hPa', 'wind_direction_850hPa', 'geopotential_height_850hPa', 'temperature_700hPa', 'wind_speed_700hPa', 'wind_direction_700hPa', 'geopotential_height_700hPa', 'temperature_500hPa', 'wind_speed_500hPa', 'wind_direction_500hPa', 'geopotential_height_500hPa', 'temperature_300hPa', 'wind_speed_300hPa', 'wind_direction_300hPa', 'geopotential_height_300hPa', 'temperature_250hPa', 'wind_speed_250hPa', 'wind_direction_250hPa', 'geopotential_height_250hPa', 'temperature_200hPa', 'wind_speed_200hPa', 'wind_direction_200hPa', 'geopotential_height_200hPa']
    weather_data = None
    if not weather_locations.empty:
        latitudes = ','.join(weather_locations['weather_latitude'].astype(str))
        longitudes = ','.join(weather_locations['weather_longitude'].astype(str))
        params = {'latitude': latitudes, 'longitude': longitudes, 'hourly': ','.join(weather_variables), 'forecast_days': 1}
        start_time = perf_counter()
        try:
            response = requests.get(weather_url, params=params, timeout=60)
            response.raise_for_status()
            weather_data = response.json()
            duration = perf_counter() - start_time
            weather_file = raw_path / f'openmeteo_{timestamp}.json'
            with open(weather_file, 'w', encoding='utf-8') as f:
                json.dump(weather_data, f)
            file_size_mb = weather_file.stat().st_size / (1024 * 1024)
            print('Open-Meteo Status Code:', response.status_code)
            print('Weather locations requested:', len(weather_locations))
            print('File size:', round(file_size_mb, 3), 'MB')
            print('Request duration:', round(duration, 2), 'seconds')
            print('Open-Meteo data saved to:', weather_file)
            write_extraction_log(source='Open-Meteo', status='SUCCESS', rows=len(weather_locations), file_size_mb=round(file_size_mb, 3), duration_seconds=duration, status_code=response.status_code)
        except requests.exceptions.Timeout:
            duration = perf_counter() - start_time
            print('Open-Meteo request timed out.')
            write_extraction_log(source='Open-Meteo', status='FAILED', rows=0, file_size_mb=0, duration_seconds=duration, error_message='Request timed out')
        except requests.exceptions.HTTPError as e:
            duration = perf_counter() - start_time
            status_code = response.status_code if 'response' in locals() else None
            print('Open-Meteo HTTP error:', e)
            write_extraction_log(source='Open-Meteo', status='FAILED', rows=0, file_size_mb=0, duration_seconds=duration, status_code=status_code, error_message=str(e))
        except requests.exceptions.RequestException as e:
            duration = perf_counter() - start_time
            print('Open-Meteo request failed:', e)
            write_extraction_log(source='Open-Meteo', status='FAILED', rows=0, file_size_mb=0, duration_seconds=duration, error_message=str(e))
    else:
        print('Weather extraction skipped because no aircraft locations were available.')
    aircraft_file = raw_path / 'aircraftDatabase.csv'
    if aircraft_file.exists():
        aircraft_file_size_mb = aircraft_file.stat().st_size / (1024 * 1024)
        print('Aircraft database is available.')
        print('File size:', round(aircraft_file_size_mb, 3), 'MB')
    else:
        print('WARNING: aircraftDatabase.csv was not found in data/raw/')
    print('Extraction Log:')
    if log_file.exists():
        extraction_log = pd.read_csv(log_file)
        display(extraction_log.tail(10))
    else:
        print('No extraction log found.')
    print('Files currently inside data/raw:')
    for file in sorted(raw_path.iterdir()):
        if file.is_file():
            file_size_mb = file.stat().st_size / (1024 * 1024)
            print('-', file.name, '|', round(file_size_mb, 3), 'MB')
