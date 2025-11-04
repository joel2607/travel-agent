# from langgraph.graph import StateGraph, END

# from graph.state import GraphState
# from graph.nodes import (
#     memory_aware_preferences_node,
#     memory_enhanced_planning_node,
#     execute_searches_node,
#     create_travel_plan_node,  
#     save_trip_to_memory_node
# )
# from graph.edges import should_continue


# def build_travel_planner_with_memory():
#     """Build LangGraph with MemGPT integration"""
#     graph_builder = StateGraph(GraphState)
    
#     # Add nodes
#     graph_builder.add_node("preferences", memory_aware_preferences_node)
#     graph_builder.add_node("queries", memory_enhanced_planning_node)
#     graph_builder.add_node("search", execute_searches_node)
#     graph_builder.add_node("plan", create_travel_plan_node)
#     graph_builder.add_node("save_memory", save_trip_to_memory_node)
    
#     # Set entry point
#     graph_builder.set_entry_point("preferences")
    
#     # Add conditional edges
#     graph_builder.add_conditional_edges(
#         "preferences",
#         should_continue,
#         {
#             "preferences": "preferences",
#             "queries": "queries",
#             "search": "search",
#             "plan": "plan",
#             "end": END
#         }
#     )
    
#     graph_builder.add_edge("queries", "search")
#     graph_builder.add_edge("search", "plan")
#     graph_builder.add_edge("plan", "save_memory")
#     graph_builder.add_edge("save_memory", END)
    
#     return graph_builder.compile()

    
from langgraph.graph import StateGraph, END


from graph.state import GraphState
from graph.nodes import (
    memory_aware_preferences_node,
    memory_enhanced_planning_node,
    execute_searches_node,
    create_travel_plan_node,  
    save_trip_to_memory_node
)
from graph.edges import should_continue


def build_travel_planner_with_memory():
    """Build LangGraph with MemGPT integration and memory-aware routing."""
    graph_builder = StateGraph(GraphState)
    
    # Add nodes
    graph_builder.add_node("preferences", memory_aware_preferences_node)
    graph_builder.add_node("queries", memory_enhanced_planning_node)
    graph_builder.add_node("search", execute_searches_node)
    graph_builder.add_node("plan", create_travel_plan_node)
    graph_builder.add_node("save_memory", save_trip_to_memory_node)
    
    # Set entry point
    graph_builder.set_entry_point("preferences")
    
    # === CONDITIONAL EDGE 1: From preferences ===
    # Allows looping on preferences or moving forward based on memory + extraction
    graph_builder.add_conditional_edges(
        "preferences",
        should_continue,
        {
            "preferences": "preferences",  # Loop if memory suggests gathering more info
            "queries": "queries",           # Move to query generation
            "search": "search",             # Skip queries if repeat trip detected
            "plan": "plan",                 # Skip to planning if full history found
            "end": END                      # End if user hasn't provided destination
        }
    )
    
    # === CONDITIONAL EDGE 2: From queries ===
    # Allows memory to suggest using previous queries or generating new ones
    graph_builder.add_conditional_edges(
        "queries",
        should_continue,
        {
            "queries": "queries",           # Regenerate if needed
            "search": "search",             # Execute searches
            "plan": "plan",                 # Skip if memory has recent results
            "end": END
        }
    )
    
    # === CONDITIONAL EDGE 3: From search ===
    # Allows memory to influence whether to re-search or move to planning
    graph_builder.add_conditional_edges(
        "search",
        should_continue,
        {
            "search": "search",             # Re-search if results insufficient
            "plan": "plan",                 # Create plan from results
            "end": END
        }
    )
    
    # === CONDITIONAL EDGE 4: From plan ===
    # Allows refinement or moving to memory save
    graph_builder.add_conditional_edges(
        "plan",
        should_continue,
        {
            "plan": "plan",                 # Refine plan if needed
            "save_memory": "save_memory",   # Save to memory
            "end": END
        }
    )
    
    # Final edge: Always save and end
    graph_builder.add_edge("save_memory", END)
    
    return graph_builder.compile()
