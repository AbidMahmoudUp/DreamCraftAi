from pydantic import BaseModel
from typing import Optional, List

class WeatherDay(BaseModel):
    Date: str
    MaxTemp: Optional[float] = None
    MinTemp: Optional[float] = None
    Humidity: Optional[float] = None