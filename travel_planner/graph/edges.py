from graph.state import GraphState

def should_continue(state: GraphState) -> str:
    """Determine the next step in the flow."""
    user_messages = [m for m in state.get('messages', []) if m.get('role') == 'user']
    processed_count = state.get('processed_message_count', 0)
    
    # No user messages at all
    if not user_messages:
        return "end"
    
    # We've processed all user messages but still don't have preferences
    # This means we're waiting for MORE user input, so end this cycle
    if len(user_messages) <= processed_count:
        # All messages processed, wait for new user input
        if 'user_preferences' not in state or not state.get('user_preferences'):
            # No preferences yet but nothing new to process
            return "end"
    
    # Check for valid user_preferences (not None and has required fields)
    preferences = state.get('user_preferences')
    if not preferences or not hasattr(preferences, 'destination') or not preferences.destination:
        return "preferences"
    
    # Check for search_queries
    queries = state.get('search_queries')
    if not queries or len(queries) == 0:
        return "queries"
    
    # Check for search_results
    results = state.get('search_results')
    if not results or len(results) == 0:
        return "search"
    
    # Check for travel_plan
    plan = state.get('travel_plan')
    if not plan:
        return "plan"
    
    # Everything complete
    return "end"
