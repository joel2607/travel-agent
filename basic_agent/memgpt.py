import os
import json
import uuid
from typing import TypedDict, List, Annotated, Any
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
import chromadb

load_dotenv(override=True)


WORKING_MEMORY_FILE = "working_memory.json"
client = chromadb.PersistentClient(path="./chroma_db")
archival_collection = client.get_or_create_collection(name="archival_memory")



# --- State ---
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str




# --- Helper Functions (Unchanged) ---
def load_working_memory(session_id: str) -> List[str]:
    if not os.path.exists(WORKING_MEMORY_FILE): return []
    with open(WORKING_MEMORY_FILE, 'r') as f:
        try: data = json.load(f)
        except json.JSONDecodeError: return []
    return data.get(session_id, [])

def save_working_memory(session_id: str, facts: List[str]):
    data = {}
    if os.path.exists(WORKING_MEMORY_FILE):
        try:
            with open(WORKING_MEMORY_FILE, 'r') as f: data = json.load(f)
        except json.JSONDecodeError: data = {}
    data[session_id] = facts
    with open(WORKING_MEMORY_FILE, 'w') as f: json.dump(data, f, indent=4)




# --- Tools ---
@tool
def add_to_working_context(session_id: str, fact: str) -> str:
    """Adds a fact to the persistent working context."""
    print(f"--- TOOL: Adding to working context: '{fact}' ---")
    facts = load_working_memory(session_id)
    if fact not in facts: facts.append(fact)
    save_working_memory(session_id, facts)
    return f"Successfully added to working context: '{fact}'."
@tool
def search_archive(query: str) -> str:
    """Performs a semantic search on the long-term vector database."""
    print(f"--- TOOL: Searching archive for: '{query}' ---")
    results = archival_collection.query(query_texts=[query], n_results=2)
    retrieved_docs = results.get('documents', [[]])[0]
    if not retrieved_docs: return "No relevant facts found in the archive."
    return "Found relevant facts:\n- " + "\n- ".join(retrieved_docs)



tools = [add_to_working_context, search_archive]
llm_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
tool_node = ToolNode(tools=tools)




# --- Agent Node ---
def agent_node(state: GraphState) -> dict[str, Any]:
    """
    This single node decides everything:
    1. Answer directly from context if possible.
    2. Call a tool if new info is provided or a search is needed.
    3. Confirm that a tool has been used.
    """
    print("--- AGENT: Processing... ---")
    session_id = state['session_id']
    facts = load_working_memory(session_id)
    working_context_str = "\n".join([f"- {fact}" for fact in facts])

    system_prompt = f"""
You are a helpful and concise AI assistant with a memory system.

**Your Top Priority: Answer the user's question directly if you can.**
Review the conversation history and the Working Context. If the answer is there, provide it immediately and concisely. **DO NOT use a tool if you already know the answer.**

**If you cannot answer directly, you MUST use a tool:**
- If the user is giving you new information (preferences, facts, names), use `add_to_working_context`.
- If the user is asking a question you don't know the answer to, use `search_archive`.

**After a tool is used**, provide a brief, simple confirmation. (e.g., "Okay, I've saved that.")

CURRENT WORKING CONTEXT:
{working_context_str}

SESSION ID FOR TOOLS: "{session_id}"
"""
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}




# --- Router ---
def router(state: GraphState) -> str:
    """A simple router. If the agent called a tool, run the tool. Otherwise, the turn is over."""
    if state["messages"][-1].tool_calls:
        return "tools"
    return END



# --- The Graph Definition ---
graph_builder = StateGraph(GraphState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.set_entry_point("agent")
graph_builder.add_edge("tools", "agent")
graph_builder.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools",
        "__end__": END 
    }
)
graph = graph_builder.compile()




# --- The Chat Loop  ---
session_id = f"session_{uuid.uuid4()}"
current_state = {"messages": [], "session_id": session_id}
print("Starting assistant. Type 'quit' or 'exit' to end.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Assistant: Goodbye!")
        break

    current_state["messages"].append(HumanMessage(content=user_input))

    final_message_content = ""
    for step in graph.stream(current_state):
        node, output = list(step.items())[0]
        # Always update the full state
        current_state["messages"].extend(output.get("messages", []))
        
        # Capture the last content message before the graph ends
        if node == "agent" and not output.get("messages", [{}])[-1].tool_calls:
            last_msg = output.get("messages", [{}])[-1]
            if last_msg and last_msg.content:
                final_message_content = last_msg.content

    if final_message_content:
        print(f"Assistant: {final_message_content}\n")