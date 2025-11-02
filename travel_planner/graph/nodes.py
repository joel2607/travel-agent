from langchain_google_genai import ChatGoogleGenerativeAI

from tools.mcp_client import MCPClient
from config.settings import settings
from graph.state import GraphState
from models.preferences import PreferencesModel, SearchQuery
from memory.memgpt_system import MemGPTSystem
from models.places import PlaceResult, TravelPlan

import os, datetime, json

def travel_preferences_node(state: GraphState):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
    llm_structured = llm.with_structured_output(PreferencesModel)

    user_messages = [m for m in state['messages'] if m.get('role') == 'user']
    
    if not user_messages:
        # First run: greet the user
        greeting = "Hi! I'm your travel planning assistant. I'll help you create a personalized travel plan. Could you tell me about your travel preferences? For example:\n- Where would you like to go?\n- How long is your trip?\n- What's your budget like?\n- Who are you traveling with?\n- What are you interested in?"
        state['messages'].append({"role": "assistant", "content": greeting})
        return state
    else:
        # Check if we have enough information to extract preferences
        conversation_text = "\n".join([m.get('content', '') for m in state['messages']])
        
        try:
            preferences = llm_structured.invoke([
                {"role": "system", "content": "Extract travel preferences from the conversation. If some information is missing, make reasonable defaults."},
                {"role": "user", "content": conversation_text}
            ])
            state['user_preferences'] = preferences
            state['messages'].append({
                "role": "assistant",
                "content": f"Perfect! I understand you want to visit {preferences.destination} for {preferences.duration} with a {preferences.budget} budget. Let me start planning your trip!"
            })
        except Exception as e:
            state['messages'].append({
                "role": "assistant",
                "content": "I need a bit more information. Could you tell me your destination and what you're interested in doing?"
            })

    return state

def generate_search_queries_node(state: GraphState):
    """Generates a strategic list of search queries based on user preferences."""
    if 'user_preferences' not in state:
        return state
    
    print("--- GENERATING SEARCH QUERIES ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3, api_key=os.getenv("GEMINI_API_KEY"))
    structured_llm = llm.with_structured_output(SearchQuery)
    
    preferences = state['user_preferences']

    query_generator_prompt = f"""
    Create 6-8 strategic Google Maps search queries for {preferences.destination} based on these preferences:
    
    - Duration: {preferences.duration}
    - Budget: {preferences.budget}
    - Companions: {preferences.companions}
    - Interests: {', '.join(preferences.interests)}
    - Must-See: {', '.join(preferences.must_see) if preferences.must_see else 'None'}
    
    Generate diverse queries covering:
    - Top attractions matching their interests
    - Restaurants fitting their budget
    - Activities suitable for their companions
    - Must-see items they mentioned
    
    Make queries specific to the destination and prioritize based on their stated interests.
    """

    try:
        search_queries_model = structured_llm.invoke(query_generator_prompt)
        search_queries = search_queries_model.queries
        state['search_queries'] = search_queries
        
        state['messages'].append({
            "role": "assistant", 
            "content": f"I've created {len(search_queries)} targeted searches to find the best spots for you. Let me search for places now..."
        })
        print(f"Generated {len(search_queries)} queries")

    except Exception as e:
        print(f"Error generating search queries: {e}")
        state['messages'].append({
            "role": "assistant",
            "content": f"I had trouble creating the search plan. Let me try a different approach."
        })

    return state

def execute_searches_node(state: GraphState):
    """Execute the search queries using the MCP server."""
    if 'search_queries' not in state or 'user_preferences' not in state:
        return state
    
    print("--- EXECUTING SEARCHES ---")
    
    mcp_client = MCPClient()
    preferences = state['user_preferences']
    all_results = []
    
    # First, geocode the destination to get coordinates for location-based searches
    destination_coords = mcp_client.geocode(preferences.destination)
    
    for query in state['search_queries']:
        print(f"Searching: {query.query} (Priority: {query.priority})")
        
        # Search for places
        places = mcp_client.search_places(
            query.query, 
            location=destination_coords if destination_coords else None,
            radius=10000  # 10km radius
        )
        
        # Convert to our PlaceResult model
        for place in places[:5]:  # Limit to top 5 results per query
            try:
                place_result = PlaceResult(
                    name=place.get('name', ''),
                    formatted_address=place.get('formatted_address', ''),
                    location=place.get('location', {}),
                    place_id=place.get('place_id', ''),
                    rating=place.get('rating'),
                    types=place.get('types', []),
                    category=query.category,
                    priority=query.priority
                )
                all_results.append(place_result)
            except Exception as e:
                print(f"Error processing place result: {e}")
                continue
    
    state['search_results'] = all_results
    
    state['messages'].append({
        "role": "assistant",
        "content": f"Great! I found {len(all_results)} places across different categories. Let me create your personalized travel plan..."
    })
    
    print(f"Found {len(all_results)} total places")
    return state

def create_travel_plan_node(state: GraphState):
    """Compile search results into a comprehensive travel plan."""
    if 'search_results' not in state or 'user_preferences' not in state:
        return state
    
    print("--- CREATING TRAVEL PLAN ---")
    
    preferences = state['user_preferences']
    results = state['search_results']
    
    # Group places by category
    places_by_category = {}
    for place in results:
        if place.category not in places_by_category:
            places_by_category[place.category] = []
        places_by_category[place.category].append(place)
    
    # Sort each category by rating and priority
    for category in places_by_category:
        places_by_category[category].sort(
            key=lambda x: (x.priority * 2 + (x.rating or 0)), 
            reverse=True
        )
    
    # Create the travel plan
    travel_plan = TravelPlan(
        destination=preferences.destination,
        total_places=len(results),
        places_by_category=places_by_category,
        recommendations=[]
    )
    
    state['travel_plan'] = travel_plan
    
    # Generate a formatted response
    plan_text = f"# Your Personalized Travel Plan for {preferences.destination}\n\n"
    plan_text += f"**Trip Duration:** {preferences.duration}\n"
    plan_text += f"**Budget:** {preferences.budget}\n"
    plan_text += f"**Traveling with:** {preferences.companions}\n\n"
    
    for category, places in places_by_category.items():
        if places:
            plan_text += f"## {category}\n"
            for i, place in enumerate(places[:3], 1):  # Top 3 per category
                rating_text = f" ({place.rating}⭐)" if place.rating else ""
                plan_text += f"{i}. **{place.name}**{rating_text}\n"
                plan_text += f"   📍 {place.formatted_address}\n\n"
    
    plan_text += "\n💡 **Tips:**\n"
    plan_text += f"- This plan is tailored for your {preferences.budget} budget and {', '.join(preferences.interests)} interests\n"
    plan_text += f"- All locations are in or near {preferences.destination}\n"
    plan_text += "- Consider checking opening hours and making reservations where needed\n"
    
    state['messages'].append({
        "role": "assistant",
        "content": plan_text
    })
    
    return state


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
    structured_llm = llm.with_structured_output(SearchQuery)
    
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