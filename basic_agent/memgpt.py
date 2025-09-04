import os
from typing import TypedDict, Annotated, List, Union, Optional
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import asyncio

from dotenv import load_dotenv

load_dotenv(override=True)

# --- Configuration and Setup ---
# NOTE: This code requires you to have the following libraries installed:
# pip install -U langchain langgraph memgpt langchain-google-genai
#
# You must set your environment variable for the Google API Key.
# export GOOGLE_API_KEY="your_api_key_here"
#
# MemGPT requires a configuration file. You can create one by running:
# memgpt configure

# Define the model to be used. The user requested "Gemini 2.5 flash lite".
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
os.environ["MEMGPT_MODEL"] = GEMINI_MODEL_NAME

# --- Define the Agent State ---
# This is the state that will be passed between nodes in the graph.
# 'messages' will store the conversation history, and 'memgpt_agent_state'
# will hold the state of our MemGPT agent.
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[List[Union[HumanMessage, AIMessage]], add_messages]
    memgpt_agent_state: dict

# --- Agent Node ---
# This node will handle the core logic of our MemGPT agent.
# It receives the current state and returns a new state with the agent's response.
def memgpt_node(state: AgentState) -> dict:
    """
    This node invokes the MemGPT agent to get a response.
    """
    print("\n--- Invoking MemGPT Agent ---")
    
    # Retrieve the last message from the conversation history.
    user_input = state['messages'][-1].content
    
    # Pass the input to the MemGPT agent.
    # We use a simple `run` method for demonstration. In a real app,
    # you would handle more complex interaction logic.
    try:
        memgpt_agent = state['memgpt_agent_state']
        response = memgpt_agent.step(user_input)
        
        # The response from MemGPT is a string. We convert it to an AIMessage.
        new_messages = [AIMessage(content=response.text)]
        
        # The MemGPT agent's internal state needs to be updated.
        # This is a simplification; a more robust solution would handle
        # state serialization and deserialization.
        state['memgpt_agent_state'] = memgpt_agent
        
        return {"messages": new_messages}
    except Exception as e:
        print(f"Error during MemGPT invocation: {e}")
        return {"messages": [AIMessage(content="I am sorry, an error occurred while processing your request.")]}

# --- Graph Definition ---
def create_agent_graph():
    """
    Creates and compiles the LangGraph state machine.
    """
    # Instantiate the LLM.
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME)
    
    # Set up the MemGPT agent configuration.
    # This is a basic configuration. For more advanced features like persona,
    # you would adjust the parameters here.
    agent_config = {
        "persona": "A helpful assistant that remembers our conversations.",
        "human_name": "User",
    }
    
    # Create the MemGPT agent instance. This is a one-time setup.
    memgpt_agent = create_memgpt_autogen_agent_from_config(agent_config)

    # Define the graph with our custom state.
    workflow = StateGraph(AgentState)

    # Add the MemGPT node to the graph.
    workflow.add_node("memgpt_agent", memgpt_node)

    # Set the entry point of the graph.
    workflow.set_entry_point("memgpt_agent")
    
    # Set the end point, which is the same as the entry point in this
    # simple conversational loop.
    workflow.add_edge("memgpt_agent", END)

    # Compile the graph.
    graph = workflow.compile()
    
    return graph, memgpt_agent

# --- Main Execution Loop ---
async def main():
    print(f"Initializing agent with model: {GEMINI_MODEL_NAME}...")
    
    # Create the graph and the initial MemGPT agent instance.
    graph, memgpt_agent = create_agent_graph()

    print("Agent is ready. Type 'exit' to quit.")

    # Main chat loop.
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            break
        
        # Prepare the initial state for the graph.
        # The agent state is passed into the graph.
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "memgpt_agent_state": memgpt_agent
        }

        # Run the graph and stream the output.
        async for s in graph.astream(initial_state):
            # Print the response from the last node in the graph.
            # In our simple case, this will be the MemGPT agent's output.
            if "__end__" in s:
                last_message = s["__end__"]["messages"][-1]
                print(f"Agent: {last_message.content}")

if __name__ == "__main__":
    asyncio.run(main())
