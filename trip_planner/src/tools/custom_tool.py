from datetime import datetime, timedelta
from crewai.tools import tool
from typing import List, Type, Union
from pydantic import BaseModel, Field
from src.pydantic_models.TravelItinerary import TravelItinerary

@tool("add_actual_dates_in_itinerary")
def add_actual_dates_in_itinerary(
    travel_itinerary: Union[TravelItinerary, dict]
) -> dict:
    """
    Updates the 'date' field in each day's plan based on the trip's start_date.
    The start_date should be in dd-mm-yyyy format.
    Accepts either a TravelItinerary object or a dictionary.
    Returns a dictionary with updated itinerary.
    """
    try:
        if isinstance(travel_itinerary, dict):
            travel_itinerary = TravelItinerary(**travel_itinerary)

        base_date = datetime.strptime(travel_itinerary.start_date, "%d-%m-%Y")

        for daily_plan in travel_itinerary.itinerary:
            offset = daily_plan.day - 1
            actual_date = (base_date + timedelta(days=offset)).strftime("%d-%m-%Y")
            daily_plan.date = actual_date

        return travel_itinerary.dict()

    except Exception as e:
        return {"error": f"Failed to add dates to itinerary: {str(e)}"}

# Add custom_tool to get hotel recommendations. 