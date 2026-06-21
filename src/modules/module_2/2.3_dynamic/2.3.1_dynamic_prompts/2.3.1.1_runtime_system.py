from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class Context:
    user_role: str
    deployment_env: str

@dynamic_prompt
def runtime_aware_prompt(request: ModelRequest[Context]) -> str:
    # Read from Runtime Context: user role and environment
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "You are a helpful assistant."

    if user_role == "admin":
        base += "\nYou have admin access. You can perform all operations. greet the user and let them know that you are an admin."
    elif user_role == "viewer":
        base += "\nYou have read-only access. Guide users to read operations only. Let the user know that you are a viewer."

    if env == "production":
        base += "\nBe extra careful with any data modifications. let the user know that you are in production."

    return base.strip()

model = init_chat_model(model='gpt-5-nano')

runtime_prompt = create_agent(
    model=model,
    tools=[],
    middleware=[runtime_aware_prompt],
    context_schema=Context
)