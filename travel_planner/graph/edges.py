from graph.state import GraphState

def should_continue(state: GraphState) -> str:
    """Determines the next step based on whether the profile is complete."""
    preferences = state.get('user_preferences')
    
    if preferences and preferences.ready_to_plan:
        # Profile is complete and user wants to plan a trip
        if not state.get('search_queries'):
            return "queries"
        if not state.get('search_results'):
            return "search"
        if not state.get('travel_plan'):
            return "plan"
        return "end"
    
    # If not ready to plan, loop back to the profiling node
    return "preferences"
