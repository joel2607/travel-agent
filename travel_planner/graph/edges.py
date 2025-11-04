# from graph.state import GraphState

# def should_continue(state: GraphState) -> str:
#     """Determine the next step in the flow."""
#     user_messages = [m for m in state.get('messages', []) if m.get('role') == 'user']
#     processed_count = state.get('processed_message_count', 0)
    
#     # No user messages at all
#     if not user_messages:
#         return "end"
    
#     # We've processed all user messages but still don't have preferences
#     # This means we're waiting for MORE user input, so end this cycle
#     if len(user_messages) <= processed_count:
#         # All messages processed, wait for new user input
#         if 'user_preferences' not in state or not state.get('user_preferences'):
#             # No preferences yet but nothing new to process
#             return "end"
    
#     # Check for valid user_preferences (not None and has required fields)
#     preferences = state.get('user_preferences')
#     if not preferences or not hasattr(preferences, 'destination') or not preferences.destination:
#         return "preferences"
    
#     # Check for search_queries
#     queries = state.get('search_queries')
#     if not queries or len(queries) == 0:
#         return "queries"
    
#     # Check for search_results
#     results = state.get('search_results')
#     if not results or len(results) == 0:
#         return "search"
    
#     # Check for travel_plan
#     plan = state.get('travel_plan')
#     if not plan:
#         return "plan"
    
#     # Everything complete
#     return "end"


from graph.state import GraphState
from memory.memgpt_system import MemGPTSystem
import json


def should_continue(state: GraphState) -> str:
    """Determine the next step in the flow with memory integration."""
    user_messages = [m for m in state.get('messages', []) if m.get('role') == 'user']
    processed_count = state.get('processed_message_count', 0)
    memgpt_system = state.get('memgpt_system')
    
    print(f"🔄 Router check: {len(user_messages)} messages, processed: {processed_count}")
    
    # No user messages at all
    if not user_messages:
        return "end"
    
    # All messages processed and waiting for new input
    if len(user_messages) <= processed_count:
        if 'user_preferences' not in state or not state.get('user_preferences'):
            print("⏸️ Waiting for more user input")
            return "end"
    
    # === MEMORY INTEGRATION POINT 1: Pre-populate preferences from memory ===
    preferences = state.get('user_preferences')
    if not preferences or not hasattr(preferences, 'destination') or not preferences.destination:
        # Before requesting preferences, check memory for similar recent trips
        if memgpt_system and len(user_messages) >= 1:
            try:
                latest_user_msg = user_messages[-1]['content']
                print(f"🧠 Searching memory for context: '{latest_user_msg[:50]}...'")
                
                # Search memory for relevant trips
                memory_results = memgpt_system.memory_store.search_archival(
                    latest_user_msg, 
                    page_size=3
                )
                
                if memory_results and len(memory_results) > 0:
                    print(f"✅ Found {len(memory_results)} relevant past trips in memory")
                    
                    # Add memory context to state for preferences node to use
                    state['memory_suggestions'] = memory_results
                    
                    # Optionally: If a very similar recent trip exists, offer quick re-planning
                    memory_context = json.dumps(memory_results, indent=2)
                    state['messages'].append({
                        "role": "assistant",
                        "content": f"I found these relevant past trips in my memory:\n{memory_context}\n\nWould you like me to recreate a plan based on these, or plan something new?"
                    })
            except Exception as e:
                print(f"⚠️ Memory search failed: {e}")
        
        print("➡️ Moving to preferences extraction")
        return "preferences"
    
    # === MEMORY INTEGRATION POINT 2: Enhance preference extraction with memory ===
    # (This happens in the preferences node, but router can flag it)
    print("✅ Valid preferences found")
    
    # Check for search_queries
    queries = state.get('search_queries')
    if not queries or len(queries) == 0:
        # Before generating queries, check memory for similar destination searches
        if memgpt_system and preferences:
            try:
                dest_query = f"{preferences.destination} {' '.join(preferences.interests)}"
                print(f"🧠 Searching memory for similar destination searches")
                
                similar_searches = memgpt_system.memory_store.search_archival(
                    dest_query,
                    page_size=2
                )
                
                if similar_searches:
                    state['memory_previous_searches'] = similar_searches
                    print(f"✅ Found {len(similar_searches)} previous searches for similar destinations")
            except Exception as e:
                print(f"⚠️ Could not retrieve previous searches: {e}")
        
        print("➡️ Generating search queries")
        return "queries"
    
    # Check for search_results
    results = state.get('search_results')
    if not results or len(results) == 0:
        print("➡️ Executing searches")
        return "search"
    
    # === MEMORY INTEGRATION POINT 3: Before plan creation, retrieve user preferences ===
    plan = state.get('travel_plan')
    if not plan:
        # Before creating plan, retrieve user's long-term travel style preferences
        if memgpt_system:
            try:
                print("🧠 Retrieving user travel preferences from long-term memory")
                
                # Query for user's general travel patterns
                user_preferences = memgpt_system.memory_store.search_archival(
                    "travel style preferences pace budget accommodation",
                    page_size=5
                )
                
                if user_preferences:
                    state['user_travel_patterns'] = user_preferences
                    print(f"✅ Retrieved user travel patterns for plan optimization")
            except Exception as e:
                print(f"⚠️ Could not retrieve travel patterns: {e}")
        
        print("➡️ Creating optimized travel plan")
        return "plan"
    
    # Everything complete
    print("✅ Flow complete, ending")
    return "end"


# Extended conditional router with memory-aware branching
def should_continue_with_memory_branches(state: GraphState) -> str:
    """
    Advanced router that uses memory to decide between different planning strategies.
    """
    memgpt_system = state.get('memgpt_system')
    preferences = state.get('user_preferences')
    
    # First call the basic router
    next_step = should_continue(state)
    
    # If we're about to plan, check if this is a "repeat trip" scenario
    if next_step == "plan" and memgpt_system and preferences:
        try:
            # Search for identical or very similar trips
            trip_key = f"{preferences.destination} {preferences.duration}"
            repeat_trips = memgpt_system.memory_store.search_archival(
                trip_key,
                page_size=1
            )
            
            if repeat_trips:
                state['is_repeat_trip'] = True
                state['previous_plan_reference'] = repeat_trips[0]
                print("🔁 Detected repeat trip - will reference previous plan")
                # Could create variant routing here, e.g., return "adapt_existing_plan"
        except Exception as e:
            print(f"⚠️ Could not check for repeat trips: {e}")
    
    return next_step
