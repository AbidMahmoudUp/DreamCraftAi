from fastapi import APIRouter, HTTPException
from typing import List
from models.weather import WeatherDay
from services.weather_service import WeatherService
from config.settings import Settings
from utils.logging import setup_logging

router = APIRouter()
settings = Settings()
logger = setup_logging()
weather_service = WeatherService(settings)

@router.get("/weather", response_model=List[WeatherDay])
async def get_weather(country: str):
    try:
        return weather_service.get_full_weather_data(country.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")