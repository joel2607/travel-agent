
import aiohttp
import json

class McpClient:
    def __init__(self, mcp_server_url: str, api_key: str = None):
        self.mcp_server_url = mcp_server_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._session = None

    async def initialize(self):
        """Initializes the aiohttp session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(headers=self.headers)

    async def close(self):
        """Closes the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def list_tools(self) -> list:
        """Lists the available tools on the MCP server."""
        await self.initialize()
        request_payload = {
            "jsonrpc": "2.0",
            "method": "listTools",
            "id": 1,
        }
        async with self._session.post(self.mcp_server_url, data=json.dumps(request_payload)) as response:
            response.raise_for_status()
            if response.content_type == 'text/event-stream':
                async for line in response.content:
                    if line.startswith(b'data:'):
                        try:
                            data = json.loads(line[5:])
                            if 'result' in data and 'tools' in data['result']:
                                return data['result']['tools']
                        except json.JSONDecodeError:
                            pass
                return []
            else:
                json_response = await response.json()
                if 'result' in json_response and 'tools' in json_response['result']:
                    return json_response['result']['tools']
                return []

    async def call_tool(self, tool_name: str, **kwargs) -> dict:
        """Calls a tool on the MCP server."""
        await self.initialize()
        request_payload = {
            "jsonrpc": "2.0",
            "method": "callTool",
            "params": {
                "name": tool_name,
                "arguments": kwargs,
            },
            "id": 1,
        }
        async with self._session.post(self.mcp_server_url, data=json.dumps(request_payload)) as response:
            response.raise_for_status()
            if response.content_type == 'text/event-stream':
                async for line in response.content:
                    if line.startswith(b'data:'):
                        try:
                            data = json.loads(line[5:])
                            if 'result' in data:
                                return data['result']
                        except json.JSONDecodeError:
                            pass
                return {}
            else:
                json_response = await response.json()
                if 'result' in json_response:
                    return json_response['result']
                return json_response

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
