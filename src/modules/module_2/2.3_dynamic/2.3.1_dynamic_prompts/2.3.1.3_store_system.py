from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore

model = init_chat_model(model='gpt-5-nano')

@dataclass
class Context:
    user_id: str

@dynamic_prompt
async def store_aware_prompt(request: ModelRequest[Context]) -> str:
    user_id = request.runtime.context.user_id

    # Read from Store: get user preferences
    store = request.runtime.store
    user_prefs = await store.aget(("preferences",), user_id)

    base = "You are a helpful assistant."

    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."
    else:
        base += "User prefers angry responses."

    return base

# store is commented out since langGraph server handles it internally
store_prompt = create_agent(
    model=model,
    middleware=[store_aware_prompt],
    context_schema=Context,
    # store=InMemoryStore(),
)
