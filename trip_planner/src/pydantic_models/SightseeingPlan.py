from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class SightSeeingPlan(BaseModel):
    plan: Dict[str, List[str]] = Field(
        ...,
        description="Dictionary where the key is a location and the value is a list of sightseeing spots for that location"
    )