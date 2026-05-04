"""Router architecture — Multiple agents variant (Send / parallel fan-out).

PARALLEL ROUTING (Send)
------------------------
Unlike the single-agent variant (Command → one agent), Send fans out to MULTIPLE
agents simultaneously when a query spans more than one domain. All selected agents
run in parallel; their results are collected and synthesized into one final answer.

Use this when:
  - A query naturally spans multiple knowledge domains.
  - You want to query all relevant sources and let a synthesizer merge results.
  - Latency matters — parallel execution is faster than sequential.

Flow:
  query → classify (returns list of domains)
        → Send to each matching agent in parallel
        → all agents write their answer to state["answers"]
        → synthesize node merges answers into one response

This demo: a product knowledge base with three vertical agents:
  - docs_agent     : official documentation and how-tos
  - changelog_agent: recent releases and breaking changes
  - community_agent: community tips, workarounds, and FAQs
"""

from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# State — shared across all nodes in the graph
# ---------------------------------------------------------------------------

class RouterState(TypedDict):
    query: str
    # Each parallel agent appends its answer here.
    # Annotated[list, operator.add] tells LangGraph to merge lists from parallel branches.
    answers: Annotated[list[str], lambda a, b: a + b]
    final_answer: str


# Per-agent state — what each parallel branch receives via Send
class AgentState(TypedDict):
    query: str
    answers: Annotated[list[str], lambda a, b: a + b]


# ---------------------------------------------------------------------------
# Classification schema — router can select multiple domains at once
# ---------------------------------------------------------------------------

class Classification(BaseModel):
    """Which agents should handle this query (one or more)."""

    domains: list[str]  # subset of: docs, changelog, community
    reason: str


# ---------------------------------------------------------------------------
# Specialized agents
# ---------------------------------------------------------------------------

model = init_chat_model(model="gpt-4o")

docs_agent = create_agent(
    model,
    system_prompt=(
        "You are a documentation expert. Answer questions using official docs, "
        "API references, and getting-started guides. Be precise and cite examples."
    ),
)

changelog_agent = create_agent(
    model,
    system_prompt=(
        "You are a release notes expert. Answer questions about recent changes, "
        "new features, deprecations, and breaking changes across versions."
    ),
)

community_agent = create_agent(
    model,
    system_prompt=(
        "You are a community knowledge expert. Share known workarounds, "
        "common pitfalls, and tips from the community and support forums."
    ),
)

_AGENTS = {
    "docs": docs_agent,
    "changelog": changelog_agent,
    "community": community_agent,
}


# ---------------------------------------------------------------------------
# Router node — classifies query into one or more domains, then fans out.
# Returns a list of Send objects — LangGraph executes all of them in parallel.
# ---------------------------------------------------------------------------

classifier = model.with_structured_output(Classification)

def router(state: RouterState) -> list[Send]:
    """Classify the query and fan out to all relevant agents in parallel."""
    result = classifier.invoke([
        {"role": "system", "content": (
            "Classify which knowledge sources should answer this query. "
            "Choose one or more from: docs, changelog, community.\n"
            "docs = official documentation, APIs, how-tos.\n"
            "changelog = recent releases, new features, breaking changes.\n"
            "community = workarounds, tips, known issues, FAQs.\n"
            "Return all that are relevant."
        )},
        {"role": "user", "content": state["query"]},
    ])

    valid_domains = [d for d in result.domains if d in _AGENTS]

    # Send dispatches each agent as an independent parallel branch.
    # Each branch receives its own copy of the state.
    return [Send(f"{domain}_agent", {"query": state["query"], "answers": []}) for domain in valid_domains]


# ---------------------------------------------------------------------------
# Agent node wrappers — run the agent and append its answer to shared state
# ---------------------------------------------------------------------------

def _make_agent_node(agent: Any, label: str):
    def node(state: AgentState) -> dict:
        result = agent.invoke({"messages": [HumanMessage(content=state["query"])]})
        answer = f"[{label}]\n{result['messages'][-1].content}"
        return {"answers": [answer]}
    node.__name__ = f"{label}_agent"
    return node

docs_node = _make_agent_node(docs_agent, "docs")
changelog_node = _make_agent_node(changelog_agent, "changelog")
community_node = _make_agent_node(community_agent, "community")


# ---------------------------------------------------------------------------
# Synthesize node — merges all parallel answers into one coherent response
# ---------------------------------------------------------------------------

synthesizer = create_agent(
    model,
    system_prompt=(
        "You are a synthesis expert. You receive answers from multiple knowledge "
        "sources and combine them into a single, well-structured response. "
        "Remove duplicates, resolve contradictions, and present a clear final answer."
    ),
)

def synthesize(state: RouterState) -> dict:
    """Merge all parallel agent answers into one final response."""
    combined = "\n\n".join(state["answers"])
    prompt = f"Original question: {state['query']}\n\nSource answers:\n{combined}"
    result = synthesizer.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"final_answer": result["messages"][-1].content}


# ---------------------------------------------------------------------------
# Graph — router fans out via Send; all branches converge at synthesize
# ---------------------------------------------------------------------------

graph = StateGraph(RouterState)

graph.add_node("router", router)
graph.add_node("docs_agent", docs_node)
graph.add_node("changelog_agent", changelog_node)
graph.add_node("community_agent", community_node)
graph.add_node("synthesize", synthesize)

graph.add_edge(START, "router")
# No explicit edges from router — Send handles the fan-out dynamically.
for agent in ("docs_agent", "changelog_agent", "community_agent"):
    graph.add_edge(agent, "synthesize")
graph.add_edge("synthesize", END)

parallel_router = graph.compile()


if __name__ == "__main__":
    result = parallel_router.invoke(
        {"query": "What changed in the latest release and are there known migration issues?", "answers": []}
    )
    print(result["final_answer"])
