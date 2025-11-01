from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class CoreMemory(BaseModel):
    """Persistent user facts - always in context"""
    user_id: str
    persona: str = "I am a helpful travel planning assistant."
    user_profile: str = "No user information yet."
    travel_style: Optional[str] = None
    budget_range: Optional[str] = None
    dietary_restrictions: List[str] = []
    accessibility_needs: List[str] = []
    preferred_accommodation: Optional[str] = None

class ConversationMessage(BaseModel):
    """Message in conversation history"""
    role: str  # user, assistant, function, system
    content: str
    timestamp: str
    metadata: Dict[str, Any] = {}

class TripMemory(BaseModel):
    """Archival memory of completed trips"""
    trip_id: str
    destination: str
    dates: str
    places_visited: List[str]
    satisfaction_score: Optional[float] = None
    highlights: List[str] = []
    lessons_learned: List[str] = []
    total_cost: Optional[float] = None
    companions: str
    embedding: Optional[List[float]] = None

class MemorySearchResult(BaseModel):
    """Result from memory search"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str  # 'recall' or 'archival'