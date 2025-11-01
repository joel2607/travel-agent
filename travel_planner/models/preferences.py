from pydantic import BaseModel, Field
from typing import List

class PreferencesModel(BaseModel):
    destination: str = Field(..., description="City or place user wants to visit")
    duration: str = Field(..., description="Length of the trip")
    budget: str = Field(..., description="Budget level: budget-friendly, mid-range, luxury")
    companions: str = Field(..., description="Travel companions: solo, friends, family, couple")
    interests: List[str] = Field(..., description="User's interests like history, shopping, adventure, etc.")
    pace: str = Field(None, description="Preferred pace: relaxed, moderate, fast-paced")
    must_see: List[str] = Field(None, description="Specific sights or activities user must see/do")

class SearchQuery(BaseModel):
    """Represents a single, strategic query to be executed."""
    category: str = Field(..., description="A high-level category for the search, e.g., 'Restaurants', 'Attractions'.")
    query: str = Field(..., description="The specific, optimized search string for Google Maps.")
    priority: int = Field(..., description="A priority score from 1-5, where 5 is most important, based on user interests.")
