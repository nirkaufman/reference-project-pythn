from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

# Initialize models once outside the middleware
premium_model = init_chat_model("gpt-5")
standard_model = init_chat_model("gpt-4o")
budget_model = init_chat_model("gpt-4-mini")

@dataclass
class Context:
    cost_tier: str
    environment: str

# Read from Runtime Context: cost tier and environment
@wrap_model_call
def context_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    cost_tier = request.runtime.context.cost_tier
    environment = request.runtime.context.environment

    if environment == "production" and cost_tier == "premium":
        model = premium_model
    elif cost_tier == "budget":
        model = budget_model
    else:
        model = standard_model

    request = request.override(model=model)

    return handler(request)

runtime_model = create_agent(
    model=budget_model,
    tools=[],
    middleware=[context_based_model],
    context_schema=Context
)