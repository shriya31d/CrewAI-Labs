from typing import List, Optional
from pydantic import BaseModel, Field
from src.pydantic_models.DailyPlan import DailyPlan

class TravelItinerary(BaseModel):
    destination: str = Field(..., description="Travel destination (e.g., Kashmir, India)")
    start_date: str = Field(..., description="Start date of the trip in DD-MM-YYYY format")
    duration_days: int = Field(..., description="Total number of days for the trip")
    traveler_age: Optional[int] = Field(None, description="Age of the traveler for personalization")
    season: Optional[str] = Field(None, description="Inferred season based on the start date and destination")
    itinerary: List[DailyPlan] = Field(..., description="List of daily plans for the trip")