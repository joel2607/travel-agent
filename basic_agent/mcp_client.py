import requests
import json
from typing import TypedDict, List, Dict, Any
# ---------------------
# MCP CLIENT
# ---------------------
class MCPClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
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
            response.raise_for_status()
            result = response.json()
            print(f"Response: {result}")
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")
            
            return result.get("result", {})
        except Exception as e:
            print(f"Error calling MCP tool {tool_name}: {e}")
            return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}
    
    def search_places(self, query: str, location: Dict[str, float] = None, radius: int = 5000) -> List[Dict]:
        """Search for places using the MCP server."""
        args = {"query": query}
        if location:
            args["location"] = location
            args["radius"] = radius
        
        result = self.call_tool("maps_search_places", args)
        
        if result.get("isError"):
            print(f"Search failed for query '{query}': {result}")
            return []
        
        try:
            content = result["content"][0]["text"]
            data = json.loads(content)
            return data.get("places", [])
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing search results: {e}")
            return []
    
    def geocode(self, address: str) -> Dict[str, float]:
        """Geocode an address to get coordinates."""
        result = self.call_tool("maps_geocode", {"address": address})
        
        if result.get("isError"):
            print(f"Geocoding failed for '{address}': {result}")
            return {}
        
        try:
            content = result["content"][0]["text"]
            data = json.loads(content)
            return data.get("location", {})
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing geocoding results: {e}")
            return {}
