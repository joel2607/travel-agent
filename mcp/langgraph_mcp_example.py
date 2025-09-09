
import os
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from mcp import McpClient

# 1. Define the state for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Create the MCP client
async def create_mcp_client():
    """Creates and initializes an MCP client."""
    client = McpClient(
        mcp_server_url="http://localhost:8000/mcp",
        # You may need to provide an API key if the server requires it
        # api_key=os.environ.get("MCP_API_KEY"),
    )
    await client.initialize()
    return client

# 3. Define the ToolNode
async def tool_node(state: State, mcp_client: McpClient) -> dict:
    """Executes tool calls using the MCP client."""
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"Calling tool: {tool_name} with args: {tool_args}")
        try:
            response = await mcp_client.call_tool(tool_name, **tool_args)
            tool_messages.append(
                ToolMessage(content=str(response), tool_call_id=tool_call["id"])
            )
        except Exception as e:
            tool_messages.append(
                ToolMessage(content=f"Error calling tool {tool_name}: {e}", tool_call_id=tool_call["id"])
            )
    return {"messages": tool_messages}

# 4. Define the graph
llm = ChatOpenAI(model="gpt-4o")

def should_continue(state: State) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

async def call_model(state: State) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(State)
graph_builder.add_node("llm", call_model)
graph_builder.add_conditional_edge("llm", should_continue)
graph_builder.add_node("tools", lambda state: tool_node(state, mcp_client))
graph_builder.add_edge("tools", "llm")
graph_builder.set_entry_point("llm")

# 5. Run the graph
async def main():
    global mcp_client
    mcp_client = await create_mcp_client()

    graph = graph_builder.compile()

    # List the available tools from the MCP server
    tools = await mcp_client.list_tools()
    print("Available tools:", tools)

    # Now, let's run the graph with a prompt that uses one of the tools
    inputs = {
        "messages": [
            (
                "user",
                "What is the geocode for the address 1600 Amphitheatre Parkway, Mountain View, CA?",
            )
        ]
    }
    async for event in graph.astream(inputs, stream_mode="values"):
        print(event)

if __name__ == "__main__":
    import asyncio
    # This is needed to run the async main function
    asyncio.run(main())