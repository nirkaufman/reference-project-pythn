from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from typing import Callable

simple_model = init_chat_model(model='gpt-5-nano')
complex_model = init_chat_model(model='gpt-4o')

# run on each model call and choose the appropriate model
@wrap_model_call
async def state_based_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if len(request.messages) > 5:
        chosen_model = complex_model
        print("Using complex model")
    else:
        chosen_model = simple_model
        print("Using simple model")

    return await handler(request.override(model=chosen_model))


state_model = create_agent(simple_model,
                           middleware=[state_based_model])
