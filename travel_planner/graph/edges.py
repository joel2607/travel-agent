from graph.state import GraphState

def should_continue(state: GraphState) -> str:
    """Determine the next step in the flow."""
    user_messages = [m for m in state.get('messages', []) if m.get('role') == 'user']
    if not user_messages:
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
