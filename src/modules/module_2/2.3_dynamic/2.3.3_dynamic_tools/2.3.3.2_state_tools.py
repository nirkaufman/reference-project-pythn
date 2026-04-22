from langchain.agents import AgentState
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from typing import Callable

@tool(description="public search function")
def public_search(query: str) -> str:
    return f"public searching for '{query}'..."

@tool(description="advanced search function")
def advanced_search(query: str) -> str:
    return f"advanced searching for '{query}'..."

@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    is_authenticated = request.state.get("authenticated", False)
    message_count = len(request.state["messages"])

    # Only enable sensitive tools after authentication
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # Limit tools early in conversation
        tools = [t for t in request.tools if t.name.startswith != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)


model = init_chat_model(model='gpt-5-nano')

class CustomState(AgentState):
    authenticated: bool


state_tools = create_agent(model=model,
                             tools=[public_search, advanced_search],
                             state_schema=CustomState,
                             middleware=[state_based_tools])
