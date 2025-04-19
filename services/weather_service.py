import requests
import json
from datetime import datetime
from typing import Dict, List
from config.settings import Settings
from utils.logging import setup_logging

logger = setup_logging()

class WeatherService:
    def __init__(self, settings: Settings):
        self.base_url = settings.weather_api_base_url
        self.country_coords = {
    "afghanistan": {"lat": 34.5553, "lon": 69.2075, "timezone": "Asia/Kabul"},
    "albania": {"lat": 41.3275, "lon": 19.8187, "timezone": "Europe/Tirane"},
    "algeria": {"lat": 36.7372, "lon": 3.0863, "timezone": "Africa/Algiers"},
    "argentina": {"lat": -34.6037, "lon": -58.3816, "timezone": "America/Argentina/Buenos_Aires"},
    "australia": {"lat": -35.2809, "lon": 149.1300, "timezone": "Australia/Canberra"},
    "bangladesh": {"lat": 23.8103, "lon": 90.4125, "timezone": "Asia/Dhaka"},
    "brazil": {"lat": -15.8267, "lon": -47.9218, "timezone": "America/Sao_Paulo"},
    "canada": {"lat": 45.4215, "lon": -75.6972, "timezone": "America/Toronto"},
    "chad": {"lat": 12.1348, "lon": 15.0557, "timezone": "Africa/Ndjamena"},
    "chile": {"lat": -33.4489, "lon": -70.6693, "timezone": "America/Santiago"},
    "china": {"lat": 39.9042, "lon": 116.4074, "timezone": "Asia/Shanghai"},
    "colombia": {"lat": 4.7110, "lon": -74.0721, "timezone": "America/Bogota"},
    "egypt": {"lat": 30.0444, "lon": 31.2357, "timezone": "Africa/Cairo"},
    "france": {"lat": 48.8566, "lon": 2.3522, "timezone": "Europe/Paris"},
    "germany": {"lat": 52.5200, "lon": 13.4050, "timezone": "Europe/Berlin"},
    "ghana": {"lat": 5.6037, "lon": -0.1870, "timezone": "Africa/Accra"},
    "greece": {"lat": 37.9838, "lon": 23.7275, "timezone": "Europe/Athens"},
    "india": {"lat": 28.6139, "lon": 77.2090, "timezone": "Asia/Kolkata"},
    "indonesia": {"lat": -6.2088, "lon": 106.8456, "timezone": "Asia/Jakarta"},
    "iran": {"lat": 35.6892, "lon": 51.3890, "timezone": "Asia/Tehran"},
    "iraq": {"lat": 33.3152, "lon": 44.3661, "timezone": "Asia/Baghdad"},
    "italy": {"lat": 41.9028, "lon": 12.4964, "timezone": "Europe/Rome"},
    "japan": {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
    "kenya": {"lat": -1.2921, "lon": 36.8219, "timezone": "Africa/Nairobi"},
    "mexico": {"lat": 19.4326, "lon": -99.1332, "timezone": "America/Mexico_City"},
    "morocco": {"lat": 33.9716, "lon": -6.8498, "timezone": "Africa/Casablanca"},
    "nigeria": {"lat": 9.0765, "lon": 7.3986, "timezone": "Africa/Lagos"},
    "norway": {"lat": 59.9139, "lon": 10.7522, "timezone": "Europe/Oslo"},
    "pakistan": {"lat": 33.6844, "lon": 73.0479, "timezone": "Asia/Karachi"},
    "peru": {"lat": -12.0464, "lon": -77.0428, "timezone": "America/Lima"},
    "philippines": {"lat": 14.5995, "lon": 120.9842, "timezone": "Asia/Manila"},
    "poland": {"lat": 52.2297, "lon": 21.0122, "timezone": "Europe/Warsaw"},
    "portugal": {"lat": 38.7223, "lon": -9.1393, "timezone": "Europe/Lisbon"},
    "russia": {"lat": 55.7558, "lon": 37.6173, "timezone": "Europe/Moscow"},
    "saudi arabia": {"lat": 24.7136, "lon": 46.6753, "timezone": "Asia/Riyadh"},
    "south africa": {"lat": -25.7479, "lon": 28.2293, "timezone": "Africa/Johannesburg"},
    "south korea": {"lat": 37.5665, "lon": 126.9780, "timezone": "Asia/Seoul"},
    "spain": {"lat": 40.4168, "lon": -3.7038, "timezone": "Europe/Madrid"},
    "sudan": {"lat": 15.5007, "lon": 32.5599, "timezone": "Africa/Khartoum"},
    "sweden": {"lat": 59.3293, "lon": 18.0686, "timezone": "Europe/Stockholm"},
    "thailand": {"lat": 13.7563, "lon": 100.5018, "timezone": "Asia/Bangkok"},
    "tunisia": {"lat": 36.8065, "lon": 10.1815, "timezone": "Africa/Tunis"},
    "turkey": {"lat": 39.9334, "lon": 32.8597, "timezone": "Europe/Istanbul"},
    "uganda": {"lat": 0.3476, "lon": 32.5825, "timezone": "Africa/Kampala"},
    "ukraine": {"lat": 50.4501, "lon": 30.5234, "timezone": "Europe/Kyiv"},
    "united kingdom": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"},
    "united states": {"lat": 38.9072, "lon": -77.0369, "timezone": "America/New_York"},
    "vietnam": {"lat": 21.0278, "lon": 105.8342, "timezone": "Asia/Ho_Chi_Minh"},
    "zambia": {"lat": -15.3875, "lon": 28.3228, "timezone": "Africa/Lusaka"}
}

    async def fetch_weather_summary(self, country: str) -> str:
        if country not in self.country_coords:
            return "Country not supported"
        try:
            lat = self.country_coords[country]["lat"]
            lon = self.country_coords[country]["lon"]
            timezone = self.country_coords[country]["timezone"]
            start_date = "2020-01-01"
            end_date = datetime.now().strftime("%Y-%m-%d")
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean&timezone={timezone}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            max_temps = [day["temperature_2m_max"] for day in data["daily"] if day["temperature_2m_max"] is not None]
            min_temps = [day["temperature_2m_min"] for day in data["daily"] if day["temperature_2m_min"] is not None]
            humidities = [day["relative_humidity_2m_mean"] for day in data["daily"] if day["relative_humidity_2m_mean"] is not None]
            summary = {
                "avg_max_temp": sum(max_temps) / len(max_temps) if max_temps else None,
                "avg_min_temp": sum(min_temps) / len(min_temps) if min_temps else None,
                "avg_humidity": sum(humidities) / len(humidities) if humidities else None
            }
            return json.dumps(summary)
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return "Weather data unavailable"

    def get_full_weather_data(self, country: str) -> List[Dict]:
        if country not in self.country_coords:
            raise ValueError("Country not supported")
        lat = self.country_coords[country]["lat"]
        lon = self.country_coords[country]["lon"]
        timezone = self.country_coords[country]["timezone"]
        start_date = "2020-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean&timezone={timezone}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "Date": data["daily"]["time"][i],
                "MaxTemp": data["daily"]["temperature_2m_max"][i],
                "MinTemp": data["daily"]["temperature_2m_min"][i],
                "Humidity": data["daily"]["relative_humidity_2m_mean"][i]
            }
            for i in range(len(data["daily"]["time"]))
        ]