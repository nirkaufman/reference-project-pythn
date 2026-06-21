"""Client that connects to the local concert MCP server and builds an agent from its tools.

Start the server first (see 3.1.2_concert_server.py), then this file can be
loaded by `langgraph dev` / Studio like any other graph module.
"""

import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

SYSTEM_PROMPT = """
    You are a concert booking assistant.
    Help users search for concerts and book tickets using the available tools.
""".strip()

mcp_client = MultiServerMCPClient(
    {
        "concert_server": {
            "transport": "http",
            "url": "http://127.0.0.1:8000/mcp",
        }
    }
)


async def _load_mcp_context():
    tools = await mcp_client.get_tools()
    resources = await mcp_client.get_resources("concert_server")
    prompt = await mcp_client.get_prompt(
        "concert_server", "concert_assistant", arguments={"query": "What's playing?"}
    )
    return tools, resources, prompt


tools, resources, prompt_messages = asyncio.run(_load_mcp_context())

print(f"Loaded {len(tools)} tool(s) from concert_server: {[t.name for t in tools]}")
print(f"Loaded resource concerts://catalog:\n{resources[0].as_string()}")
print(f"Loaded prompt 'concert_assistant':\n{prompt_messages}")

model = init_chat_model(model="gpt-4o")

concert_agent = create_agent(
    model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)
