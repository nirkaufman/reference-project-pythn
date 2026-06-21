"""Travel agent that uses tools from a remote MCP server."""

import asyncio
from datetime import date

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

SYSTEM_PROMPT = f"""
    You are a travel assistant powered by Kiwi flight search.

    Rules:
    - Today is {date.today().strftime("%d/%m/%Y")}
    - Always use future dates in dd/mm/yyyy format
    - Convert relative dates ("next week", "in March") to actual dates
    - Use IATA airport codes when possible (e.g., JFK, LHR, CDG)
    - If a date is ambiguous or in the past, ask for clarification
""".strip()

# Connect to the remote Kiwi travel MCP server via HTTP
mcp_client = MultiServerMCPClient(
    {
        "travel_server": {
            "transport": "http",
            "url": "https://mcp.kiwi.com",
        }
    }
)

# Load tools from the remote MCP server
tools = asyncio.run(mcp_client.get_tools())

model = init_chat_model(model="gpt-4o")

travel_agent = create_agent(
    model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)
