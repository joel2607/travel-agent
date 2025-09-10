
import os
import asyncio
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import McpClient

import functools

# 1. Define the state for the graph
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Define the ToolNode
async def tool_node(state: State, mcp_client: McpClient) -> dict:
    """Executes tool calls using the MCP client."""
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"Calling tool: {tool_name} with args: {tool_args}")
        try:
            # The response from call_tool is already a dict, so no need to parse it as JSON
            response = await mcp_client.call_tool(tool_name, **tool_args)
            tool_messages.append(
                ToolMessage(content=str(response), tool_call_id=tool_call["id"])
            )
        except Exception as e:
            tool_messages.append(
                ToolMessage(content=f"Error calling tool {tool_name}: {e}", tool_call_id=tool_call["id"])
            )
    return {"messages": tool_messages}

# 3. Define the graph
def should_continue(state: State) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

async def call_model(state: State, llm: ChatGoogleGenerativeAI
                     ) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# 4. Run the graph
async def main():
    """Main function to run the LangGraph application."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        api_key="AIzaSyDMtrQ2dH82CL8CfWYWzaIhOP0qz5Ta0VE"
    )

    async with McpClient(mcp_server_url="http://localhost:8000/mcp") as mcp_client:
        # List the available tools from the MCP server
        tools = await mcp_client.list_tools()
        print("Available tools:", tools)

        graph_builder = StateGraph(State)
        graph_builder.add_node("llm", functools.partial(call_model, llm=llm))
        graph_builder.add_conditional_edges("llm", should_continue)
        graph_builder.add_node("tools", functools.partial(tool_node, mcp_client=mcp_client))
        graph_builder.add_edge("tools", "llm")
        graph_builder.set_entry_point("llm")

        graph = graph_builder.compile()

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
    # This is needed to run the async main function
    asyncio.run(main())