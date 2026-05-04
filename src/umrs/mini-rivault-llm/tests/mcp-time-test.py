import asyncio

from fastmcp import Client

# Standard MCP configuration
config = {
    "mcpServers": {
        "time": {
            "url": "http://llm.ai.r-ccs.riken.jp:11434/time/mcp",
            "headers": {
                "x-litellm-api-key": "Bearer sk-ll-324kmjh8"
            }
        }
    }
}

# Create a client that connects to the server
client = Client(config)

async def main():
    async with client:
        # List available tools generated from OpenAPI spec
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        # Example: Get a pet by ID (from Petstore API)
        response = await client.call_tool(name="current_time")
        print(f"Response:\n{response}\n")

if __name__ == "__main__":
    asyncio.run(main())
