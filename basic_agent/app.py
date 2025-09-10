from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
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


# ---------------------
# Graph State
# ---------------------
class GraphState(TypedDict):
    messages: List[BaseMessage]
    user_preferences: PreferencesModel
    search_queries: List[SearchQuery]

# ---------------------
# Collect Preferences
# ---------------------
def travel_preferences_node(state: GraphState):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
    llm_structured = llm.with_structured_output(PreferencesModel)

    user_messages = [m for m in state['messages'] if m['role'] == 'user']
    
    if not user_messages:
        # First run: include a placeholder user message
        greeting_prompt = [
            {"role": "system", "content": (
                "You are a friendly travel assistant. "
                "Greet the user and ask them about their travel preferences."
            )},
            {"role": "user", "content": "Hi"}  # placeholder to satisfy Gemini
        ]
        greeting = llm.invoke(greeting_prompt)
        state['messages'].append({"role": "assistant", "content": greeting.content})
    else:
        # Collect structured preferences
        preferences = llm_structured.invoke(state['messages'])
        state['user_preferences'] = preferences
        state['messages'].append({
            "role": "assistant",
            "content": f"Preferences collected: {preferences.model_dump()}"
        })

    return state


# ---------------------
# Generate Smart Search Queries
# ---------------------
def generate_search_queries_node(state: GraphState):
    """Generates a strategic list of search queries based on user preferences."""
    print("--- GENERATING SEARCH QUERIES ---")
    
    # Use your preferred LLM setup
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3, api_key=os.getenv("GEMINI_API_KEY"))
    
    # We want the LLM to output a list of SearchQuery objects
    structured_llm = llm.with_structured_output(SearchQueries)
    
    preferences = state['user_preferences']

    # This prompt is key. It guides the LLM to think like a planner.
    query_generator_prompt = f"""
    You are an expert travel planner. Based on the user's preferences below, create a strategic list of 5 to 8 targeted Google Maps search queries to find the best locations for their trip.

    **User Preferences:**
    - Destination: {preferences.destination}
    - Interests: {', '.join(preferences.interests)}
    - Budget: {preferences.budget}
    - Companions: {preferences.companions}
    - Must-See: {', '.join(preferences.must_see) if preferences.must_see else 'None'}

    **Your Task:**
    Generate queries that are specific and cover different categories (like food, attractions, activities). 
    Assign a priority (5=highest) based on how closely a query matches the user's stated interests and must-see items.

    **Example:** For a user interested in 'history' and 'food' on a 'budget' in 'Rome', you might generate:
    - category: "Attractions", query: "historical landmarks and ancient ruins in Rome", priority: 5
    - category: "Restaurants", query: "top-rated budget-friendly trattorias in Rome", priority: 5
    - category: "Activities", query: "guided tours of the Colosseum and Roman Forum", priority: 4
    """

    # Invoke the LLM to get the structured search plan
    try:
        search_queries_model = structured_llm.invoke(query_generator_prompt)
        search_queries = search_queries_model.queries
        state['search_queries'] = search_queries
        
        # Add a message to the state to show progress
        state['messages'].append({
            "role": "assistant", 
            "content": f"I've created a search plan with {len(search_queries)} queries to find the best spots in {preferences.destination}."
        })
        print(f"Successfully generated {len(search_queries)} queries.")

    except Exception as e:
        print(f"Error generating search queries: {e}")
        state['messages'].append({
            "role": "assistant",
            "content": f"I had trouble creating a search plan. Error: {str(e)}"
        })
        # You might want to handle this more gracefully, but for now, we'll just report it.

    return state


# ---------------------
# Build Graph
# ---------------------
graph_builder = StateGraph(GraphState)
graph_builder.add_node("travel_preferences", travel_preferences_node)
graph_builder.add_node("generate_queries", generate_search_queries_node)
graph_builder.set_entry_point("travel_preferences")
graph_builder.add_edge("travel_preferences", "generate_queries")
graph_builder.add_edge("generate_queries", END)

graph = graph_builder.compile()

# ---------------------
# Run Graph
# ---------------------
print("Starting travel assistant...")
inputs = {
    'messages': []
}

while True:
    for s in graph.stream(inputs):
        # Print the last assistant message if it exists
        messages = s['travel_preferences']['messages']
        if messages and messages[-1]['role'] == 'assistant':
            print(messages[-1]['content'])

    x = input("You: ")
    if x.lower() in ['exit', 'quit']:
        break

    inputs['messages'].append({"role": "user", "content": x})
