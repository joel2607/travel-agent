from pydantic import BaseModel
from typing import List, Dict

class PlaceResult(BaseModel):
    """Represents a place found from search."""
    name: str
    formatted_address: str
    location: Dict[str, float]  # lat, lng
    place_id: str
    rating: float = None
    types: List[str] = []
    category: str  # From our search query
    priority: int  # From our search query

class TravelPlan(BaseModel):
    """Final compiled travel plan."""
    destination: str
    total_places: int
    places_by_category: Dict[str, List[PlaceResult]]
    recommendations: List[str]