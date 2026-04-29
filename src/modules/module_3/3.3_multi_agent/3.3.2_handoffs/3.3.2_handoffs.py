"""Handoff architecture demo: Customer support pipeline.

HANDOFFS vs SUPERVISOR
----------------------
Supervisor (see 3.3.1): A central agent decides which sub-agent to call next.
  Routing is decided by the model on every turn. Sub-agents are tools; control
  always returns to the supervisor. Best for: parallel tasks, open-ended delegation.

Handoffs (this file): A SINGLE agent changes its own behavior based on state.
  A tool call triggers a state transition ("handoff") that permanently alters the
  agent's system prompt and available tools on the next turn. There is no central
  router — the agent hands itself off to the next "role". Best for: sequential
  workflows where each step must unlock the next (e.g. collect data before acting).

HOW THE HANDOFF WORKS HERE
---------------------------
State holds `current_step` (triage → classify → resolve).
Transition tools return a Command that writes the new step into state.
The middleware reads `current_step` before every model call and injects the
matching system prompt + tool subset — so the agent literally becomes a different
specialist at each stage of the conversation.

Flow: triage (warranty?) → classify (issue type?) → resolve
  in_warranty  + software → troubleshooting steps
  in_warranty  + hardware → warranty repair
  no_warranty  + software → troubleshooting steps
  no_warranty  + hardware → escalate to human
"""

from typing import Callable, Literal

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command


# ---------------------------------------------------------------------------
# State — the single source of truth that drives all handoffs
# ---------------------------------------------------------------------------

class SupportState(AgentState):
    """Conversation state. current_step is the handoff signal read by middleware."""

    current_step: str = "triage"
    warranty_status: Literal["in_warranty", "no_warranty"] | None = None
    issue_type: Literal["software", "hardware"] | None = None


# ---------------------------------------------------------------------------
# Handoff tools — writing to state IS the handoff
# Each tool returns a Command that updates current_step, triggering middleware
# to reconfigure the agent before the next model call.
# ---------------------------------------------------------------------------

@tool
def record_warranty_status(
    status: Literal["in_warranty", "no_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record warranty status and hand off to issue classification."""
    return Command(update={
        "messages": [ToolMessage(content=f"Warranty: {status}", tool_call_id=runtime.tool_call_id)],
        "warranty_status": status,
        "current_step": "classify",  # <-- handoff
    })


@tool
def record_issue_type(
    issue_type: Literal["software", "hardware"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record issue type and hand off to resolution."""
    return Command(update={
        "messages": [ToolMessage(content=f"Issue: {issue_type}", tool_call_id=runtime.tool_call_id)],
        "issue_type": issue_type,
        "current_step": "resolve",  # <-- handoff
    })


# ---------------------------------------------------------------------------
# Resolution tools (leaf actions — no further handoffs)
# ---------------------------------------------------------------------------

@tool(description="Provide software troubleshooting steps to the customer.")
def provide_troubleshooting_steps(issue_description: str) -> str:
    return (
        f"Troubleshooting steps for: {issue_description}\n"
        "1. Restart the device.\n"
        "2. Check for software updates.\n"
        "3. Clear app cache.\n"
        "4. Reinstall the app if the issue persists."
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
        "A specialist will contact you within 24 hours. Case ID: #CS-2026-001."
    )


# ---------------------------------------------------------------------------
# Step configs — each step defines a focused persona + restricted tool set
# ---------------------------------------------------------------------------

def _get_config(step: str, warranty: str | None, issue: str | None) -> dict:
    if step == "triage":
        return {
            "prompt": "You are a triage agent. Ask if the device is under warranty, then call record_warranty_status.",
            "tools": [record_warranty_status],
        }
    if step == "classify":
        return {
            "prompt": "Ask whether the issue is software (crashes, bugs) or hardware (screen, battery). Call record_issue_type.",
            "tools": [record_issue_type],
        }
    # resolve — outcome depends on collected state
    if issue == "hardware" and warranty == "no_warranty":
        return {
            "prompt": "This hardware issue has no warranty coverage. Escalate using escalate_to_human.",
            "tools": [escalate_to_human],
        }
    if issue == "hardware":
        return {
            "prompt": "The device is under warranty. Initiate a repair using initiate_warranty_repair.",
            "tools": [initiate_warranty_repair],
        }
    return {
        "prompt": "Provide software troubleshooting steps using provide_troubleshooting_steps.",
        "tools": [provide_troubleshooting_steps],
    }


# ---------------------------------------------------------------------------
# Middleware — intercepts every model call to apply the current step's config
# This is what makes the handoff take effect: same agent, different behavior.
# ---------------------------------------------------------------------------

@wrap_model_call
async def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Swap system prompt and tools based on current_step before each model call."""
    config = _get_config(
        request.state.get("current_step", "triage"),
        request.state.get("warranty_status"),
        request.state.get("issue_type"),
    )
    return await handler(request.override(
        system_prompt=config["prompt"],
        tools=config["tools"],
    ))


# ---------------------------------------------------------------------------
# Agent — one agent, all tools registered, middleware drives the handoffs
# ---------------------------------------------------------------------------

model = init_chat_model(model="gpt-4o")

support_agent = create_agent(
    model,
    tools=[
        record_warranty_status,
        record_issue_type,
        provide_troubleshooting_steps,
        initiate_warranty_repair,
        escalate_to_human,
    ],
    state_schema=SupportState,  # type: ignore[arg-type]
    middleware=[apply_step_config],
)
