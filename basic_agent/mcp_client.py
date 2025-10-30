# mcp_client.py (Corrected Version)

import requests
import json
from typing import List, Dict, Any

class MCPClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server and correctly parse the SSE response."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        print(f"""
            payload: {json.dumps(payload)}
              """)

        try:
            response = requests.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                timeout=30
            )
            print(f"Response: {response.text}")
            response.raise_for_status()

            # --- START OF THE CRITICAL FIX ---
            # The server responds with Server-Sent Events (SSE), not raw JSON.
            # We must find the line with the JSON data and parse it manually.
            response_text = response.text

            # Find the line that starts with 'data: '
            data_line = next((line for line in response_text.splitlines() if line.startswith('data: ')), None)

            if not data_line:
                raise ValueError("No 'data:' line found in the server's SSE response")

            # Remove the 'data: ' prefix to get the clean JSON string
            json_str = data_line[len('data: '):]
            
            # Parse the clean JSON string
            parsed_response = json.loads(json_str)
            # --- END OF THE CRITICAL FIX ---

            if "error" in parsed_response:
                raise Exception(f"MCP Error: {parsed_response['error']}")

            # The actual tool result is nested within the 'result' key
            return parsed_response.get("result", {})

        except Exception as e:
            print(f"Error calling MCP tool {tool_name}: {e}")
            return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}

    def search_places(self, query: str, location: Dict[str, float] = None, radius: int = 10000) -> List[Dict]:
        """Search for places using the MCP server."""
        args = {"query": query}
        if location:
            args["location"] = f"{location['lat']},{location['lng']}" # Pass location as a string for compatibility
            args["radius"] = radius

        result = self.call_tool("maps_search_places", args)

        if result.get("isError"):
            print(f"Search failed for query '{query}': {result}")
            return []

        try:
            # The actual place data is a JSON string inside the 'text' field
            content = result["content"][0]["text"]
            data = json.loads(content)
            return data.get("places", [])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            print(f"Error parsing search results for query '{query}': {e}")
            return []

    def geocode(self, address: str) -> Dict[str, float]:
        """Geocode an address to get coordinates."""
        result = self.call_tool("maps_geocode", {"address": address})

        if result.get("isError"):
            print(f"Geocoding failed for '{address}': {result}")
            return {}

        try:
            # The actual geocode data is a JSON string inside the 'text' field
            content = result["content"][0]["text"]
            data = json.loads(content)
            # The geocode tool returns the full details, we just need the location
            if "location" in data:
                return data["location"]
            return {}
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            print(f"Error parsing geocoding results for '{address}': {e}")
            return {}