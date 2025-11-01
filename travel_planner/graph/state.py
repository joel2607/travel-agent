from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

from models.preferences import PreferencesModel, SearchQuery
from models.places import PlaceResult, TravelPlan
from memory.memgpt_system import MemGPTSystem


class GraphState(TypedDict):
    # Original fields
    messages: List[Dict[str, str]]
    user_preferences: Optional[PreferencesModel]
    search_queries: Optional[List[SearchQuery]]
    search_results: Optional[List[PlaceResult]]
    travel_plan: Optional[TravelPlan]
    
    # New memory fields
    user_id: str
    memgpt_system: Optional[MemGPTSystem]
    context_usage: Optional[int]