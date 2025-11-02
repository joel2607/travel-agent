from langchain_google_genai import ChatGoogleGenerativeAI

from tools.mcp_client import MCPClient
from config.settings import settings
from graph.state import GraphState
from models.preferences import PreferencesModel, SearchQueries
from memory.memgpt_system import MemGPTSystem
from models.places import PlaceResult, TravelPlan

import os, datetime, json


def memory_aware_preferences_node(state: GraphState) -> GraphState:
    """Extract preferences using MemGPT memory system"""
    
    # Get or create MemGPT system for this user
    if 'memgpt_system' not in state or state['memgpt_system'] is None:
        state['memgpt_system'] = MemGPTSystem(state.get('user_id', 'default_user'))
    
    memgpt = state['memgpt_system']
    
    # Track which messages we've processed to avoid loops
    if 'processed_message_count' not in state:
        state['processed_message_count'] = 0
    
    # Get user messages
    user_messages = [m for m in state.get('messages', []) if m.get('role') == 'user']
    
    # If first message, just greet
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
        state['processed_message_count'] = 0
        return state
    
    # Check if we've already processed all current user messages
    if len(user_messages) <= state['processed_message_count']:
        # No new messages to process, just return
        print("⚠️ No new user messages to process, skipping...")
        return state
    
    # Process only the latest unprocessed user message
    latest_message = user_messages[-1]['content']
    
    # Mark this message as processed BEFORE processing to prevent re-entry
    state['processed_message_count'] = len(user_messages)
    
    print(f"📝 Processing user message #{state['processed_message_count']}: {latest_message[:50]}...")
    
    # Process through MemGPT
    result = memgpt.process_message(latest_message)
    
    # Add response to messages
    state['messages'].append({
        "role": "assistant",
        "content": result['response']
    })
    
    # Only try to extract preferences if:
    # 1. We have at least 2 user messages (initial query + follow-up)
    # 2. OR the assistant response indicates readiness to plan
    # 3. AND we don't already have valid preferences
    
    should_extract = False
    
    # Check if we already have valid preferences
    existing_prefs = state.get('user_preferences')
    if existing_prefs and hasattr(existing_prefs, 'destination') and existing_prefs.destination:
        # Already have valid preferences, skip extraction
        print("✅ Already have valid preferences, skipping extraction")
        return state
    
    # Check if assistant indicated readiness
    last_assistant_msg = result['response'].lower()
    readiness_signals = [
        "let me start planning",
        "let me create your plan",
        "i'll search for places",
        "let me search for",
        "i understand you want to visit"
    ]
    
    if any(signal in last_assistant_msg for signal in readiness_signals):
        should_extract = True
        print("🎯 Detected readiness signal, will extract preferences")
    
    # Or if we have multiple exchanges
    if len(user_messages) >= 2:
        should_extract = True
        print(f"🎯 Have {len(user_messages)} user messages, will attempt extraction")
    
    if not should_extract:
        print("⏸️ Not ready to extract preferences yet")
        return state
    
    # Try to extract structured preferences from conversation
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0,
            api_key=settings.GEMINI_API_KEY
        )
        structured_llm = llm.with_structured_output(PreferencesModel)
        
        # Include core memory in extraction
        conversation_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" 
            for m in state['messages']
        ])
        core_context = f"User Profile: {memgpt.working_context.user_profile}"
        
        extraction_prompt = f"""Extract travel preferences from this conversation. 
If the user hasn't specified something, use reasonable defaults:
- duration: "1 week" 
- budget: "mid-range"
- companions: "solo"
- interests: ["sightseeing"]
- pace: "moderate"

Only extract preferences if the user has at least specified a destination.
If no destination is mentioned, return None for the destination field.

Context from memory:
{core_context}

Conversation:
{conversation_text}
"""
        
        preferences = structured_llm.invoke([
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": "Extract preferences from the above conversation"}
        ])
        
        # Validate that we have a real destination
        if preferences and preferences.destination and preferences.destination.strip():
            # Additional validation - check it's not a placeholder
            invalid_destinations = ["none", "unknown", "not specified", "n/a", ""]
            if preferences.destination.lower().strip() not in invalid_destinations:
                state['user_preferences'] = preferences
                print(f"✅ Successfully extracted preferences: {preferences.destination}")
            else:
                print(f"⚠️ Invalid destination extracted: {preferences.destination}")
        else:
            print("⚠️ No valid destination found in conversation yet")
            
    except Exception as e:
        print(f"Could not extract preferences yet: {e}")
        import traceback
        traceback.print_exc()
    
    return state


def memory_enhanced_planning_node(state: GraphState) -> GraphState:
    """Generate search queries using memory context"""
    preferences = state.get('user_preferences')
    memgpt_system = state.get('memgpt_system')
    
    # Validate we have what we need
    if not preferences:
        print("❌ No preferences found, cannot generate queries")
        return state
    
    if not hasattr(preferences, 'destination') or not preferences.destination:
        print("❌ No destination in preferences, cannot generate queries")
        return state
    
    if not memgpt_system:
        print("⚠️ No MemGPT system found, proceeding without memory context")
        memgpt = None
    else:
        memgpt = memgpt_system
    
    print(f"--- GENERATING SEARCH QUERIES for {preferences.destination} ---")
    
    # Search for similar past trips if MemGPT available
    context_str = ""
    if memgpt:
        try:
            past_trips_query = f"{preferences.destination} {' '.join(preferences.interests)}"
            past_trips = memgpt.memory_store.search_archival(past_trips_query, page_size=3)
            
            if past_trips:
                context_str = f"\n\nRelevant past trips:\n{json.dumps(past_trips, indent=2)}"
        except Exception as e:
            print(f"⚠️ Could not search past trips: {e}")
    
    # Generate queries with memory context
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0.3,
        api_key=settings.GEMINI_API_KEY
    )
    
    structured_llm = llm.with_structured_output(SearchQueries)  # <-- Changed

    user_profile = memgpt.working_context.user_profile if memgpt else "No previous history"
    
    query_prompt = f"""Create 6-8 strategic Google Maps search queries for {preferences.destination}.
    
Current trip preferences:
- Duration: {preferences.duration}
- Budget: {preferences.budget}
- Companions: {preferences.companions}
- Interests: {', '.join(preferences.interests)}
- Must-See: {', '.join(preferences.must_see) if preferences.must_see else 'None'}

User context from memory:
{user_profile}
{context_str}

Generate diverse queries covering:
- Top attractions matching their interests
- Restaurants fitting their budget
- Activities suitable for their companions
- Must-see items they mentioned

Make queries specific to the destination and prioritize based on their stated interests.

Return a JSON object with a "queries" field containing an array of objects.
Each query object should have:
- category: string (e.g., "Restaurants", "Attractions", "Activities")
- query: string (the search query for Google Maps)
- priority: integer (1-5, where 5 is most important)"""
    
    try:
        # This now returns SearchQueries object containing list of SearchQuery objects
        search_queries_wrapper = structured_llm.invoke(query_prompt)
        search_queries = search_queries_wrapper.queries  # Extract the list
        
        if search_queries and len(search_queries) > 0:
            state['search_queries'] = search_queries  # Store the list
            print(f"✅ Generated {len(search_queries)} search queries")
            
            state['messages'].append({
                "role": "assistant",
                "content": f"I've created {len(search_queries)} targeted searches to find the best spots for you. Let me search for places now..."
            })
        else:
            print("❌ No search queries generated")
            state['messages'].append({
                "role": "assistant",
                "content": "I had trouble creating search queries. Could you provide more details about what you'd like to do?"
            })
    except Exception as e:
        print(f"❌ Error generating search queries: {e}")
        import traceback
        traceback.print_exc()
        state['messages'].append({
            "role": "assistant",
            "content": "I encountered an error while planning. Let me try a different approach."
        })
    
    return state


def execute_searches_node(state: GraphState):
    """Execute the search queries using the MCP server."""
    queries = state.get('search_queries')
    preferences = state.get('user_preferences')
    
    if not queries or len(queries) == 0:
        print("❌ No search queries to execute")
        return state
        
    if not preferences:
        print("❌ No preferences available")
        return state
    
    print("--- EXECUTING SEARCHES ---")
    
    mcp_client = MCPClient()
    all_results = []
    
    # First, geocode the destination to get coordinates for location-based searches
    try:
        destination_coords = mcp_client.geocode(preferences.destination)
    except Exception as e:
        print(f"⚠️ Could not geocode destination: {e}")
        destination_coords = None
    
    for query in queries:
        try:
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
                    print(f"⚠️ Error processing place result: {e}")
                    continue
        except Exception as e:
            print(f"⚠️ Error searching for {query.query}: {e}")
            continue
    
    if len(all_results) > 0:
        state['search_results'] = all_results
        
        state['messages'].append({
            "role": "assistant",
            "content": f"Great! I found {len(all_results)} places across different categories. Let me create your personalized travel plan..."
        })
        
        print(f"✅ Found {len(all_results)} total places")
    else:
        print("❌ No search results found")
        state['messages'].append({
            "role": "assistant",
            "content": "I couldn't find any places matching your criteria. Could you provide more details or try a different destination?"
        })
    
    return state


def create_travel_plan_node(state: GraphState):
    """Compile search results into a comprehensive travel plan."""
    results = state.get('search_results')
    preferences = state.get('user_preferences')
    
    if not results or len(results) == 0:
        print("❌ No search results to create plan from")
        return state
        
    if not preferences:
        print("❌ No preferences available")
        return state
    
    print("--- CREATING TRAVEL PLAN ---")
    
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
    
    print(f"✅ Travel plan created with {len(results)} places")
    
    return state


def save_trip_to_memory_node(state: GraphState) -> GraphState:
    """Save completed trip plan to archival memory"""
    plan = state.get('travel_plan')
    memgpt_system = state.get('memgpt_system')
    
    if not plan:
        print("⚠️ No travel plan to save")
        return state
    
    if not memgpt_system:
        print("⚠️ No MemGPT system available to save memory")
        return state
    
    memgpt = memgpt_system
    
    # Create trip summary for archival
    trip_summary = f"""Destination: {plan.destination}
Total places: {plan.total_places}
Categories: {', '.join(plan.places_by_category.keys())}
Top places: {', '.join([p.name for places in plan.places_by_category.values() for p in places[:2]])}
"""
    
    try:
        # Insert into archival storage
        memgpt.memory_store.insert_archival(
            content=trip_summary,
            metadata={
                "destination": plan.destination,
                "type": "trip_plan",
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        
        print(f"✅ Saved trip plan to memory for future reference")
    except Exception as e:
        print(f"⚠️ Could not save to memory: {e}")
    
    return state
