from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes import (
    memory_aware_preferences_node,
    memory_enhanced_planning_node,
    execute_searches_node,  # Your existing node
    create_travel_plan_node,  # Your existing node
    save_trip_to_memory_node
)
from graph.edges import should_continue


def build_travel_planner_with_memory():
    """Build LangGraph with MemGPT integration"""
    graph_builder = StateGraph(GraphState)
    
    # Add nodes
    graph_builder.add_node("preferences", memory_aware_preferences_node)
    graph_builder.add_node("queries", memory_enhanced_planning_node)
    graph_builder.add_node("search", execute_searches_node)
    graph_builder.add_node("plan", create_travel_plan_node)
    graph_builder.add_node("save_memory", save_trip_to_memory_node)
    
    # Set entry point
    graph_builder.set_entry_point("preferences")
    
    # Add conditional edges
    graph_builder.add_conditional_edges(
        "preferences",
        should_continue,
        {
            "preferences": "preferences",
            "queries": "queries",
            "search": "search",
            "plan": "plan",
            "end": END
        }
    )
    
    graph_builder.add_edge("queries", "search")
    graph_builder.add_edge("search", "plan")
    graph_builder.add_edge("plan", "save_memory")
    graph_builder.add_edge("save_memory", END)
    
    return graph_builder.compile()