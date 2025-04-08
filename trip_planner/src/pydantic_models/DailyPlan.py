from typing import List, Optional
from pydantic import BaseModel, Field

class DailyPlan(BaseModel):
    day: int = Field(..., description="Day number of the itinerary")
    date: Optional[str] = Field(None, description="Actual date for this days")
    location: str = Field(..., description="Primary location or area visited on this day (e.g., Srinagar, Gulmarg)")
    spots: List[str] = Field(..., description="List of sightseeing spots for the day")