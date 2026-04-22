from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import dynamic_prompt, ModelRequest

model = init_chat_model(model='gpt-5-nano')

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    # request.messages is a shortcut for request.state[ "messages"]
    message_count = len(request.messages)

    base = "You are a helpful assistant."

    if message_count > 3:
        base += "\nThis is a long conversation - be extra concise - and let the user know that you are in a long conversation."

    return base

state_prompt = create_agent(model=model, middleware=[state_aware_prompt])
