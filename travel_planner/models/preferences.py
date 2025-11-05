from pydantic import BaseModel, Field
from typing import List

class PreferencesModel(BaseModel):
    destination: str | None = Field(default=None, description="City or place user wants to visit")
    duration: str | None = Field(default=None, description="Length of the trip")
    budget: str | None = Field(default=None, description="Budget level: budget-friendly, mid-range, luxury")
    companions: str | None = Field(default=None, description="Travel companions: solo, friends, family, couple")
    interests: List[str] | None = Field(default=None, description="User's interests like history, shopping, adventure, etc.")
    pace: str | None = Field(default=None, description="Preferred pace: relaxed, moderate, fast-paced")
    must_see: List[str] | None = Field(default=None, description="Specific sights or activities user must see/do")
    ready_to_plan: bool = Field(default=False, description="Flag to indicate when to start planning the trip.")

class SearchQuery(BaseModel):
    """Represents a single, strategic query to be executed."""
    category: str = Field(..., description="A high-level category for the search, e.g., 'Restaurants', 'Attractions'.")
    query: str = Field(..., description="The specific, optimized search string for Google Maps.")
    priority: int = Field(..., description="A priority score from 1-5, where 5 is most important, based on user interests.")

class SearchQueries(BaseModel):
    """A wrapper containing a list of search queries."""
    queries: List[SearchQuery] = Field(..., description="List of 6-8 strategic search queries")
