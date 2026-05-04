"""Router architecture — Single agent variant (Command).

ROUTER vs OTHER PATTERNS
-------------------------
Supervisor (3.3.1): The model decides routing dynamically on every turn,
  maintaining full conversation context. Best for open-ended orchestration.

Router (this file): A dedicated classification step routes the query BEFORE
  any agent runs. Routing is a preprocessing decision, not a conversational one.
  Best for clear input categories with deterministic or lightweight classification.

SINGLE AGENT ROUTING (Command)
--------------------------------
The router classifies the query into exactly ONE category and issues a Command
to jump directly to that agent's node. Use this when queries map cleanly to a
single domain (billing → billing agent, technical → support agent).

Flow: query → classify → Command(goto=agent) → agent → answer
"""

from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Classification schema — structured output forces a valid routing decision
# ---------------------------------------------------------------------------

class Classification(BaseModel):
    """Router output: which agent should handle this query."""

    domain: Literal["billing", "technical", "general"]
    reason: str


# ---------------------------------------------------------------------------
# Specialized agents — each owns a narrow domain
# ---------------------------------------------------------------------------

model = init_chat_model(model="gpt-4o")

billing_agent = create_agent(
    model,
    system_prompt=(
        "You are a billing specialist. Handle questions about invoices, "
        "subscriptions, refunds, and payment methods. Be concise and helpful."
    ),
)

technical_agent = create_agent(
    model,
    system_prompt=(
        "You are a technical support engineer. Handle questions about bugs, "
        "integrations, APIs, and configuration. Provide step-by-step guidance."
    ),
)

general_agent = create_agent(
    model,
    system_prompt=(
        "You are a general customer assistant. Handle onboarding questions, "
        "feature explanations, and anything that doesn't fit billing or technical."
    ),
)


# ---------------------------------------------------------------------------
# Router node — classifies the query and issues a Command to the right agent.
# This is a single LLM call, not a full agent — fast and deterministic.
# ---------------------------------------------------------------------------

classifier = model.with_structured_output(Classification)

def router(state: MessagesState) -> Command[Literal["billing_agent", "technical_agent", "general_agent"]]:
    """Classify the query and route to the appropriate agent."""
    result = classifier.invoke([
        {"role": "system", "content": (
            "Classify the user's query into one of: billing, technical, general. "
            "billing = invoices, payments, refunds. "
            "technical = bugs, APIs, integrations, configuration. "
            "general = everything else."
        )},
        *state["messages"],
    ])
    # Command.goto tells LangGraph which node to execute next.
    return Command(goto=f"{result.domain}_agent")


# ---------------------------------------------------------------------------
# Agent node wrappers — invoke the agent and return its messages to the graph
# ---------------------------------------------------------------------------

def run_billing_agent(state: MessagesState) -> dict:
    return billing_agent.invoke(state)

def run_technical_agent(state: MessagesState) -> dict:
    return technical_agent.invoke(state)

def run_general_agent(state: MessagesState) -> dict:
    return general_agent.invoke(state)


# ---------------------------------------------------------------------------
# Graph — router is the single entry point; agents are the leaf nodes
# ---------------------------------------------------------------------------

graph = StateGraph(MessagesState)

graph.add_node("router", router)
graph.add_node("billing_agent", run_billing_agent)
graph.add_node("technical_agent", run_technical_agent)
graph.add_node("general_agent", run_general_agent)

graph.add_edge(START, "router")
# Router uses Command(goto=...) so no explicit edges needed from router to agents.
for agent in ("billing_agent", "technical_agent", "general_agent"):
    graph.add_edge(agent, END)

single_router = graph.compile()


if __name__ == "__main__":
    result = single_router.invoke(
        {"messages": [HumanMessage(content="My invoice from last month looks wrong, I was charged twice.")]}
    )
    print(result["messages"][-1].content)
