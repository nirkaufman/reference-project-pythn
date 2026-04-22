from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

model = init_chat_model(model='gpt-5')

personal_assistant_prompt = """
    You are a personal assistant that like to chat.
    your name is: "Amanda".
    your expertise is: "brainstorming".
""".strip()

# node style hook
@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 5:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

# node style hook
@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None

custom = create_agent(
    model,
    system_prompt=personal_assistant_prompt,
    middleware=[
        check_message_limit,
        log_response
    ],
)
