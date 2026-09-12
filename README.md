# DE-SkyPrint

## Data Sources

### 1. OpenSky Network API
- **Purpose:** Live aircraft state-vector data such as aircraft identifier, position, altitude, velocity, and vertical rate.
- **Endpoint:** `https://opensky-network.org/api/states/all`
- **Authentication:** The current extraction uses anonymous access, so no credentials are required. OpenSky also supports OAuth2 client-credentials authentication for higher authenticated limits.
- **Rate limits:** OpenSky uses an API-credit system. Anonymous users receive 400 credits per day. A global `/states/all` request costs 4 credits; smaller bounding-box requests can cost fewer credits. When credits are exhausted, the API may return HTTP `429 Too Many Requests`.
- **Pagination:** Not applicable to the `/states/all` endpoint because it returns the current state-vector response in a single request.
- **Terms / licence:** OpenSky REST API data is intended for non-profit research and educational use under OpenSky's Data License Agreement. Commercial or operational use requires written permission/licensing from OpenSky.
- **Reference:** https://opensky-network.org/about/terms-of-use

### 2. Open-Meteo Forecast API
- **Purpose:** Weather enrichment data such as temperature, humidity, wind speed, wind direction, and surface pressure.
- **Endpoint:** `https://api.open-meteo.com/v1/forecast`
- **Authentication:** No API key or account is required for the free non-commercial API.
- **Rate limits:** Free non-commercial use is limited to 600 calls per minute, 5,000 calls per hour, and 10,000 calls per day.
- **Pagination:** Not applicable for the forecast request used in this project; the requested hourly values are returned in one JSON response.
- **Licence / terms:** API data is provided under Creative Commons Attribution 4.0 (CC BY 4.0). Attribution is required. The free API is for non-commercial use.
- **Reference:** https://open-meteo.com/en/terms

### 3. OpenSky Aircraft Database CSV
- **Purpose:** Reference aircraft metadata used to enrich OpenSky state vectors, including registration, manufacturer, model, and type code.
- **Source type:** CSV source file stored in `data/raw/aircraftDatabase.csv`.
- **Authentication:** Not applicable because this is a supplied/downloaded source file.
- **Rate limits:** Not applicable.
- **Licence / terms:** OpenSky states that the aircraft database is unlicensed and provided "as is"; it does not fall under the general OpenSky data terms of use.
- **Reference:** https://opensky-network.org/data/aircraft
