"""
Weather Data Ingestion from OpenWeatherMap API
"""
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert

from ..config import get_settings
from ..database.models import WeatherData, init_database

settings = get_settings()


class WeatherIngester:
    """Ingest weather data from OpenWeatherMap API"""
    
    CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
    FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
    HISTORICAL_URL = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    
    def __init__(self):
        self.api_key = settings.openweather_api_key
        self.lat = settings.chicago_lat
        self.lon = settings.chicago_lon
        
    def fetch_current_weather(self) -> Optional[Dict[str, Any]]:
        """Fetch current weather for Chicago"""
        
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not configured")
            return None
            
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'imperial'  # Fahrenheit
        }
        
        try:
            response = requests.get(self.CURRENT_WEATHER_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self._transform_current_weather(data)
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch current weather: {e}")
            return None
    
    def fetch_forecast(self) -> List[Dict[str, Any]]:
        """Fetch 5-day forecast (3-hour intervals)"""
        
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not configured")
            return []
            
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'imperial'
        }
        
        try:
            response = requests.get(self.FORECAST_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for item in data.get('list', []):
                forecasts.append(self._transform_forecast_item(item))
                
            return forecasts
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch weather forecast: {e}")
            return []
    
    def _transform_current_weather(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform current weather API response"""
        
        weather_info = data.get('weather', [{}])[0]
        main_info = data.get('main', {})
        wind_info = data.get('wind', {})
        clouds_info = data.get('clouds', {})
        rain_info = data.get('rain', {})
        snow_info = data.get('snow', {})
        
        return {
            'timestamp': datetime.fromtimestamp(data.get('dt', datetime.now().timestamp())),
            'temperature': main_info.get('temp'),
            'feels_like': main_info.get('feels_like'),
            'humidity': main_info.get('humidity'),
            'pressure': main_info.get('pressure'),
            'visibility': data.get('visibility'),
            'wind_speed': wind_info.get('speed'),
            'wind_direction': wind_info.get('deg'),
            'weather_main': weather_info.get('main'),
            'weather_description': weather_info.get('description'),
            'clouds_percentage': clouds_info.get('all'),
            'rain_1h': rain_info.get('1h', 0),
            'snow_1h': snow_info.get('1h', 0)
        }
    
    def _transform_forecast_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform forecast item"""
        
        weather_info = item.get('weather', [{}])[0]
        main_info = item.get('main', {})
        wind_info = item.get('wind', {})
        clouds_info = item.get('clouds', {})
        rain_info = item.get('rain', {})
        snow_info = item.get('snow', {})
        
        return {
            'timestamp': datetime.fromtimestamp(item.get('dt', datetime.now().timestamp())),
            'temperature': main_info.get('temp'),
            'feels_like': main_info.get('feels_like'),
            'humidity': main_info.get('humidity'),
            'pressure': main_info.get('pressure'),
            'visibility': item.get('visibility'),
            'wind_speed': wind_info.get('speed'),
            'wind_direction': wind_info.get('deg'),
            'weather_main': weather_info.get('main'),
            'weather_description': weather_info.get('description'),
            'clouds_percentage': clouds_info.get('all'),
            'rain_1h': rain_info.get('3h', 0) / 3 if rain_info.get('3h') else 0,  # Convert 3h to 1h
            'snow_1h': snow_info.get('3h', 0) / 3 if snow_info.get('3h') else 0
        }
    
    def save_to_database(self, records: List[Dict[str, Any]], database_url: str) -> int:
        """Save weather data to PostgreSQL"""
        
        if not records:
            logger.warning("No weather data to save")
            return 0
            
        engine = create_engine(database_url)
        
        with engine.begin() as conn:
            stmt = insert(WeatherData).values(records)
            stmt = stmt.on_conflict_do_nothing()
            conn.execute(stmt)
                
        logger.info(f"Saved {len(records)} weather records")
        return len(records)


def ingest_weather_data(
    database_url: Optional[str] = None,
    include_forecast: bool = True
) -> int:
    """Main function to ingest weather data"""
    
    if database_url is None:
        database_url = settings.database_url
        
    init_database(database_url)
    
    ingester = WeatherIngester()
    
    records = []
    
    # Fetch current weather
    current = ingester.fetch_current_weather()
    if current:
        records.append(current)
        logger.info("Fetched current weather")
    
    # Fetch forecast
    if include_forecast:
        forecasts = ingester.fetch_forecast()
        records.extend(forecasts)
        logger.info(f"Fetched {len(forecasts)} forecast records")
    
    if not records:
        logger.warning("No weather data fetched")
        return 0
        
    return ingester.save_to_database(records, database_url)


def generate_synthetic_weather(
    start_date: datetime,
    end_date: datetime,
    database_url: Optional[str] = None
) -> int:
    """Generate synthetic weather data for testing when API is unavailable"""
    
    import random
    import math
    
    if database_url is None:
        database_url = settings.database_url
        
    init_database(database_url)
    
    records = []
    current_date = start_date
    
    while current_date <= end_date:
        # Simulate daily temperature cycle
        hour = current_date.hour
        base_temp = 35 + 20 * math.sin((hour - 6) * math.pi / 12)  # Peak at 2pm
        
        # Add some randomness
        temp = base_temp + random.uniform(-5, 5)
        
        # Weather conditions
        weather_options = [
            ('Clear', 'clear sky', 0, 0),
            ('Clouds', 'scattered clouds', 0, 0),
            ('Clouds', 'overcast clouds', 0, 0),
            ('Rain', 'light rain', random.uniform(0.1, 0.5), 0),
            ('Rain', 'moderate rain', random.uniform(0.5, 1.5), 0),
            ('Snow', 'light snow', 0, random.uniform(0.1, 0.3))
        ]
        
        # More likely to be clear/cloudy
        weights = [0.3, 0.25, 0.2, 0.1, 0.05, 0.1]
        weather_main, weather_desc, rain, snow = random.choices(weather_options, weights=weights)[0]
        
        record = {
            'timestamp': current_date,
            'temperature': round(temp, 1),
            'feels_like': round(temp - random.uniform(2, 8), 1),
            'humidity': random.randint(40, 90),
            'pressure': random.randint(1010, 1030),
            'visibility': random.randint(5000, 10000),
            'wind_speed': round(random.uniform(0, 20), 1),
            'wind_direction': random.randint(0, 360),
            'weather_main': weather_main,
            'weather_description': weather_desc,
            'clouds_percentage': random.randint(0, 100),
            'rain_1h': rain,
            'snow_1h': snow
        }
        records.append(record)
        
        current_date += timedelta(hours=1)
    
    engine = create_engine(database_url)
    
    with engine.begin() as conn:
        for i in range(0, len(records), 1000):
            batch = records[i:i+1000]
            stmt = insert(WeatherData).values(batch)
            stmt = stmt.on_conflict_do_nothing()
            conn.execute(stmt)
    
    logger.info(f"Generated {len(records)} synthetic weather records")
    return len(records)


if __name__ == "__main__":
    count = ingest_weather_data()
    print(f"Ingested {count} weather records")
