from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from typing import Callable
from langchain_core.messages import SystemMessage

model = init_chat_model(model='gpt-5-nano')

@wrap_model_call
async def enrich_system_message(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:

    enriched_content_blocks = list(request.system_message.content_blocks) + [
        {"type": "text", "text": "You are a personal chef assistant."}
    ]

    updated_system_message= SystemMessage(content=enriched_content_blocks)
    return await handler(request.override(system_message=updated_system_message))


dynamic_system = create_agent(model=model,
                              middleware=[enrich_system_message])
