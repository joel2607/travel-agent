from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages passed between nodes.
    """
    messages: List[BaseMessage]

# Define a node that uses an LLM
def call_model(state: GraphState):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key="AIzaSyDMtrQ2dH82CL8CfWYWzaIhOP0qz5Ta0VE")
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages': [response]}

# Create the graph
graph_builder = StateGraph(GraphState)

# Add nodes to the graph
graph_builder.add_node("model_node", call_model)

# Set the entry and exit points
graph_builder.set_entry_point("model_node")
graph_builder.add_edge("model_node", END)

# Compile the graph
graph = graph_builder.compile()

# Run the graph (optional, just to show it works)
print("Running the graph...")
inputs = {
    'messages': [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ]
}

while True:


    for s in graph.stream(inputs):
        print(s['model_node']['messages'][0].content)

    x = input()

    inputs['messages'].append({"role": "user", "content": x})

    if x.lower() in ['exit', 'quit']:
        break