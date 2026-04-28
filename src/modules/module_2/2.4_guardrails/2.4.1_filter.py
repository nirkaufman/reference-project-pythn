from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import before_agent, AgentState
from langgraph.runtime import Runtime
from langchain.chat_models import init_chat_model

banned_keywords = ["hack", "exploit", "malware"]

# Deterministic guardrail: Block requests containing banned keywords.
@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # Get the first user message
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    content = first_message.content.lower()

    # Check for banned keywords
    for keyword in banned_keywords:
        if keyword in content:
            # Block execution before any processing
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                }],
                "jump_to": "end"
            }

    return None


model = init_chat_model(model='gpt-5-nano')

keyword_guardrail = create_agent(
    model=model,
    system_prompt="You are a helpful assistant named: Anna.",
    tools=[],
    middleware=[content_filter],
)
