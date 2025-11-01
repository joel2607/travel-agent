from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings
from graph.state import GraphState
from models.preferences import PreferencesModel, SearchQueries
from memory.memgpt_system import MemGPTSystem


def memory_aware_preferences_node(state: GraphState) -> GraphState:
    """Extract preferences using MemGPT memory system"""
    
    # Get or create MemGPT system for this user
    if 'memgpt_system' not in state:
        state['memgpt_system'] = MemGPTSystem(state.get('user_id', 'default_user'))
    
    memgpt = state['memgpt_system']
    
    # If first message, just greet
    user_messages = [m for m in state['messages'] if m.get('role') == 'user']
    if not user_messages:
        greeting = """Hi! I'm your travel planning assistant with memory. 
I'll remember your preferences and past trips to provide personalized recommendations.

Tell me about your travel plans:
- Where would you like to go?
- How long is your trip?
- What's your budget?
- Who are you traveling with?
- What interests you?"""
        
        state['messages'].append({"role": "assistant", "content": greeting})
        return state
    
    # Process latest user message through MemGPT
    latest_message = user_messages[-1]['content']
    result = memgpt.process_message(latest_message)
    
    # Add response to messages
    state['messages'].append({
        "role": "assistant",
        "content": result['response']
    })
    
    # Try to extract structured preferences from conversation
    if len(user_messages) >= 1:  # After at least one exchange
        try:
            llm = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL,
                temperature=0,
                api_key=settings.GEMINI_API_KEY
            )
            structured_llm = llm.with_structured_output(PreferencesModel)
            
            # Include core memory in extraction
            conversation_text = "\n".join([
                m.get('content', '') for m in state['messages']
            ])
            core_context = f"User Profile: {memgpt.working_context.user_profile}"
            
            preferences = structured_llm.invoke([
                {
                    "role": "system", 
                    "content": f"Extract travel preferences. Use defaults if needed. {core_context}"
                },
                {"role": "user", "content": conversation_text}
            ])
            
            state['user_preferences'] = preferences
            
        except Exception as e:
            print(f"Could not extract preferences yet: {e}")
    
    return state


def memory_enhanced_planning_node(state: GraphState) -> GraphState:
    """Generate search queries using memory context"""
    if 'user_preferences' not in state or 'memgpt_system' not in state:
        return state
    
    memgpt = state['memgpt_system']
    preferences = state['user_preferences']
    
    # Search for similar past trips
    past_trips_query = f"{preferences.destination} {' '.join(preferences.interests)}"
    past_trips = memgpt.memory_store.search_archival(past_trips_query, page_size=3)
    
    # Generate queries with memory context
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0.3,
        api_key=settings.GEMINI_API_KEY
    )
    structured_llm = llm.with_structured_output(SearchQueries)
    
    context_str = ""
    if past_trips:
        context_str = f"\n\nPast trips: {json.dumps(past_trips, indent=2)}"
    
    query_prompt = f"""Create 6-8 strategic Google Maps queries for {preferences.destination}.
    
Current trip preferences:
- Duration: {preferences.duration}
- Budget: {preferences.budget}
- Companions: {preferences.companions}
- Interests: {', '.join(preferences.interests)}

User context from memory:
{memgpt.working_context.user_profile}
{context_str}

Generate diverse queries that account for their past preferences and patterns."""
    
    search_queries = structured_llm.invoke(query_prompt)
    state['search_queries'] = search_queries.queries
    
    return state


def save_trip_to_memory_node(state: GraphState) -> GraphState:
    """Save completed trip plan to archival memory"""
    if 'travel_plan' not in state or 'memgpt_system' not in state:
        return state
    
    memgpt = state['memgpt_system']
    plan = state['travel_plan']
    
    # Create trip summary for archival
    trip_summary = f"""Destination: {plan.destination}
Total places: {plan.total_places}
Categories: {', '.join(plan.places_by_category.keys())}
Top places: {', '.join([p.name for places in plan.places_by_category.values() for p in places[:2]])}
"""
    
    # Insert into archival storage
    memgpt.memory_store.insert_archival(
        content=trip_summary,
        metadata={
            "destination": plan.destination,
            "type": "trip_plan",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    print(f"✅ Saved trip plan to memory for future reference")
    
    return state