from typing import Callable, Literal

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command



# Custom state

class SupportState(AgentState):
    """Tracks which step is active and what was collected so far."""
    current_step: str = "triage"
    warranty_status: Literal["in_warranty", "no_warranty"] | None = None
    issue_type: Literal["software", "hardware"] | None = None



# Transition tools

@tool
def record_warranty_status(
    status: Literal["in_warranty", "no_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and move to issue classification."""
    return Command(update={
        "messages": [ToolMessage(
            content=f"Warranty status recorded: {status}",
            tool_call_id=runtime.tool_call_id,
        )],
        "warranty_status": status,
        "current_step": "classify",
    })


@tool
def record_issue_type(
    issue_type: Literal["software", "hardware"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and move to resolution."""
    return Command(update={
        "messages": [ToolMessage(
            content=f"Issue type recorded: {issue_type}",
            tool_call_id=runtime.tool_call_id,
        )],
        "issue_type": issue_type,
        "current_step": "resolve",
    })



# Resolution tools

@tool(description="Provide software troubleshooting steps to the customer.")
def provide_troubleshooting_steps(issue_description: str) -> str:
    return (
        f"Troubleshooting steps for: {issue_description}\n"
        "1. Restart the device and try again.\n"
        "2. Check for pending software updates.\n"
        "3. Clear the application cache.\n"
        "4. Reinstall the application if the issue persists."
    )


@tool(description="Initiate a warranty repair request for a hardware issue.")
def initiate_warranty_repair(issue_description: str) -> str:
    return (
        f"Warranty repair initiated for: {issue_description}\n"
        "A prepaid shipping label has been emailed to you.\n"
        "Expected turnaround: 5–7 business days."
    )


@tool(description="Escalate the case to a human support agent.")
def escalate_to_human(issue_description: str) -> str:
    return (
        f"Escalating to human agent for: {issue_description}\n"
        "A support specialist will contact you within 24 hours.\n"
        "Your case ID is #CS-2026-001."
    )



# Step configurations

_STEP_CONFIGS = {
    "triage": {
        "prompt": (
            "You are a customer support triage agent. "
            "Your only job is to find out whether the customer's device is under warranty. "
            "Ask politely, then call record_warranty_status with 'in_warranty' or 'no_warranty'."
        ),
        "tools": [record_warranty_status],
    },
    "classify": {
        "prompt": (
            "You are a customer support agent classifying the customer's issue. "
            "Ask whether the problem is a software issue (app crashes, bugs, updates) "
            "or a hardware issue (screen, battery, physical damage). "
            "Then call record_issue_type with 'software' or 'hardware'."
        ),
        "tools": [record_issue_type],
    },
}

def _resolve_config(warranty: str | None, issue: str | None) -> dict:
    """Return the prompt and tools for the resolve step."""
    if issue == "software":
        return {
            "prompt": (
                "You are a software support specialist. "
                "Provide clear troubleshooting steps using the provide_troubleshooting_steps tool."
            ),
            "tools": [provide_troubleshooting_steps],
        }
    if issue == "hardware" and warranty == "in_warranty":
        return {
            "prompt": (
                "You are a warranty repair specialist. "
                "Initiate a warranty repair for the customer using the initiate_warranty_repair tool."
            ),
            "tools": [initiate_warranty_repair],
        }
    # hardware + no_warranty
    return {
        "prompt": (
            "You are a senior support agent. "
            "This case requires human escalation. "
            "Use the escalate_to_human tool and reassure the customer."
        ),
        "tools": [escalate_to_human],
    }


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@wrap_model_call
async def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Dynamically apply system prompt and tools based on current_step."""
    step = request.state.get("current_step", "triage")

    if step == "resolve":
        config = _resolve_config(
            request.state.get("warranty_status"),
            request.state.get("issue_type"),
        )
    else:
        config = _STEP_CONFIGS.get(step, _STEP_CONFIGS["triage"])

    request = request.override(
        system_prompt=config["prompt"],
        tools=config["tools"],
    )
    return await handler(request)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

model = init_chat_model(model="gpt-4o")

all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_troubleshooting_steps,
    initiate_warranty_repair,
    escalate_to_human,
]

support_agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config],
    # checkpointer=InMemorySaver(),
)
