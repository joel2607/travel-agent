from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp_client import MCPClient
import os

from dotenv import load_dotenv
load_dotenv(override=True)


# ---------------------
# MODELS
# ---------------------
class PreferencesModel(BaseModel):
    destination: str = Field(..., description="City or place user wants to visit")
    duration: str = Field(..., description="Length of the trip")
    budget: str = Field(..., description="Budget level: budget-friendly, mid-range, luxury")
    companions: str = Field(..., description="Travel companions: solo, friends, family, couple")
    interests: List[str] = Field(..., description="User's interests like history, shopping, adventure, etc.")
    pace: str = Field(None, description="Preferred pace: relaxed, moderate, fast-paced")
    must_see: List[str] = Field(None, description="Specific sights or activities user must see/do")

class SearchQuery(BaseModel):
    """Represents a single, strategic query to be executed."""
    category: str = Field(..., description="A high-level category for the search, e.g., 'Restaurants', 'Attractions'.")
    query: str = Field(..., description="The specific, optimized search string for Google Maps.")
    priority: int = Field(..., description="A priority score from 1-5, where 5 is most important, based on user interests.")

class SearchQueries(BaseModel):
    """A list of search queries."""
    queries: List[SearchQuery]

class PlaceResult(BaseModel):
    """Represents a place found from search."""
    name: str
    formatted_address: str
    location: Dict[str, float]  # lat, lng
    place_id: str
    rating: float = None
    types: List[str] = []
    category: str  # From our search query
    priority: int  # From our search query

class TravelPlan(BaseModel):
    """Final compiled travel plan."""
    destination: str
    total_places: int
    places_by_category: Dict[str, List[PlaceResult]]
    recommendations: List[str]

# ---------------------
# Graph State
# ---------------------
class GraphState(TypedDict):
    messages: List[BaseMessage]
    user_preferences: PreferencesModel
    search_queries: List[SearchQuery]
    search_results: List[PlaceResult]
    travel_plan: TravelPlan

# ---------------------
# Memory States
# --------------------

class CoreMemory(BaseModel):
    """Persistent user facts that rarely change"""
    user_id: str
    budget_range: str
    travel_style: str  # adventurous, relaxed, cultural
    dietary_restrictions: List[str]
    accessibility_needs: List[str]
    preferred_accommodation_type: str

class RecallMemory(BaseModel):
    """Semantic memory of past interactions"""
    memory_id: str
    content: str
    category: str
    timestamp: str
    embedding: List[float] = None

# ---------------------
# NODE FUNCTIONS
# ---------------------
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
    structured_llm = llm.with_structured_output(SearchQueries)
    
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

def create_memory_management_tools():
    """Create tools for the agent to manage its own memory"""
    
    def update_core_memory(user_id: str, field: str, value: str):
        """Update persistent user information"""
        namespace = (user_id, "core_memory")
        current = store.get(namespace, "core")
        if current:
            current.value[field] = value
            store.put(namespace, "core", current.value)
        return f"Updated {field} to {value}"
    
    def search_past_trips(user_id: str, query: str, limit: int = 5):
        """Search through past trip memories"""
        namespace = (user_id, "past_trips")
        results = store.search(namespace, query=query, limit=limit)
        return [r.value for r in results]
    
    def add_trip_memory(user_id: str, trip_data: Dict):
        """Store a completed trip for future reference"""
        namespace = (user_id, "past_trips")
        memory_id = f"trip_{datetime.now().timestamp()}"
        store.put(namespace, memory_id, trip_data)
        return "Trip memory saved"
    
    return [update_core_memory, search_past_trips, add_trip_memory]

# Add memory retrieval node
def memory_retrieval_node(state: GraphState, store):
    """Retrieve relevant memories before planning"""
    user_id = state.get('user_id', 'default_user')
    preferences = state['user_preferences']
    
    # Get core memories
    core_namespace = (user_id, "core_memory")
    core_memory = store.get(core_namespace, "core")
    
    # Search past trips for similar destinations or interests
    trip_namespace = (user_id, "past_trips")
    similar_trips = store.search(
        trip_namespace,
        query=f"{preferences.destination} {' '.join(preferences.interests)}",
        limit=3
    )
    
    # Enrich state with memories
    state['core_memory'] = core_memory.value if core_memory else {}
    state['similar_past_trips'] = [t.value for t in similar_trips]
    
    state['messages'].append({
        "role": "assistant",
        "content": f"I remember you've been to {len(similar_trips)} similar destinations before. Let me use that to personalize your plan!"
    })
    
    return state

def should_continue(state: GraphState) -> str:
    """Determine the next step in the flow."""
    user_messages = [m for m in state['messages'] if m.get('role') == 'user']
    if not user_messages:
        return "end"
        
    if 'user_preferences' not in state:
        return "preferences"
    elif 'search_queries' not in state:
        return "queries"
    elif 'search_results' not in state:
        return "search"
    elif 'travel_plan' not in state:
        return "plan"
    else:
        return "end"


# ---------------------
# BUILD GRAPH
# ---------------------
def build_travel_planner():
    graph_builder = StateGraph(GraphState)
    
    # Add nodes
    graph_builder.add_node("preferences", travel_preferences_node)
    graph_builder.add_node("queries", generate_search_queries_node)
    graph_builder.add_node("search", execute_searches_node)
    graph_builder.add_node("plan", create_travel_plan_node)
    
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
    graph_builder.add_edge("plan", END)
    
    return graph_builder.compile()

# ---------------------
# MAIN EXECUTION
# ---------------------
def main():
    print("🌍 Welcome to your AI Travel Planner!")
    print("Make sure your MCP server is running on localhost:8000")
    print("-" * 50)
    
    graph = build_travel_planner()

    #Optional: Visualize the graph (requires mermaid-cli)
    # try:
    #     graph.get_graph().draw_mermaid_png(output_file_path="travel_planner_graph.png")
    #     print("✅ Graph visualization saved to travel_planner_graph.png")
    # except Exception as e:
    #     print(f"⚠️ Could not generate graph image: {e}")
    #     print("Please ensure you have either pygraphviz or mermaid-cli installed for graph visualization.")

    # input()

    # Initialize state
    inputs = {
        'messages': []
    }
    
    while True:
        try:
            # Run the graph
            for step in graph.stream(inputs):
                for node_name, node_state in step.items():
                    messages = node_state.get('messages', [])
                    if messages and messages[-1].get('role') == 'assistant':
                        print(f"\n🤖 Assistant: {messages[-1]['content']}")
            
            # Check if we're done
            if 'travel_plan' in inputs:
                print("\n✅ Your travel plan is ready!")
                break
            
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Happy travels!")
                break
            
            if user_input:
                inputs['messages'].append({"role": "user", "content": user_input})
        
        except KeyboardInterrupt:
            print("\n👋 Travel planning interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            break

if __name__ == "__main__":
    main()