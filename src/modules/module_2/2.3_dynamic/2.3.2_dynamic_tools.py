from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from typing import Callable

model = init_chat_model(model='gpt-5-nano')
all_tools= []

@wrap_model_call
async def select_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Middleware to select relevant tools based on state/context."""
    relevant_tools = select_relevant_tools(request.state, request.runtime)
    return await handler(request.override(tools=relevant_tools))


dynamic_tools = create_agent(model=model,
                             tools=all_tools,
                             middleware=[select_tools])
