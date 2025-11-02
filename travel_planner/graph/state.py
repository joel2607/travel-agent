from typing import TypedDict, List, Dict, Any, Optional
from models.preferences import PreferencesModel, SearchQuery  # Keep SearchQuery import
from models.places import PlaceResult, TravelPlan
from memory.memgpt_system import MemGPTSystem


class GraphState(TypedDict):
    # Original fields
    messages: List[Dict[str, str]]
    user_preferences: Optional[PreferencesModel]
    search_queries: Optional[List[SearchQuery]]  # List of SearchQuery objects
    search_results: Optional[List[PlaceResult]]
    travel_plan: Optional[TravelPlan]
    
    # Memory fields
    user_id: str
    memgpt_system: Optional[MemGPTSystem]
    context_usage: Optional[int]
    
    # Loop prevention
    processed_message_count: Optional[int]
