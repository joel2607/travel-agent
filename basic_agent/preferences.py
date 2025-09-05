from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------
# Define Preferences Model
# ---------------------
class PreferencesModel(BaseModel):
    destination: str = Field(..., description="City or place user wants to visit")
    duration: str = Field(..., description="Length of the trip")
    budget: str = Field(..., description="Budget level: budget-friendly, mid-range, luxury")
    companions: str = Field(..., description="Travel companions: solo, friends, family, couple")
    interests: List[str] = Field(..., description="User's interests like history, shopping, adventure, etc.")
    pace: str = Field(None, description="Preferred pace: relaxed, moderate, fast-paced")
    must_see: List[str] = Field(None, description="Specific sights or activities user must see/do")

# ---------------------
# Define Graph State
# ---------------------
class GraphState(TypedDict):
    messages: List[BaseMessage]
    user_preferences: PreferencesModel  # Will store structured preferences

# ---------------------
# Node: Greet & Collect Preferences
# ---------------------
def travel_preferences_node(state: GraphState):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        api_key="AIzaSyDMtrQ2dH82CL8CfWYWzaIhOP0qz5Ta0VE"
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
# Build Graph
# ---------------------
graph_builder = StateGraph(GraphState)
graph_builder.add_node("travel_preferences", travel_preferences_node)
graph_builder.set_entry_point("travel_preferences")
graph_builder.add_edge("travel_preferences", END)
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
