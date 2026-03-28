# state.py
# Defines the custom agent state for the left-over meal agent.
#
# Concept: Access State (read-only from tools)
# The agent's state is shared across all tool invocations within a single run.
# Tools can READ state values via ToolRuntime[MealAgentState], but cannot write to it.
# State is initialized when the agent is invoked (e.g., from the LangGraph Studio UI).

from langchain.agents import AgentState


class MealAgentState(AgentState):
    """Extended agent state that carries the user's dietary preferences.

    This value is set once at invocation time and read by tools to
    personalize both the local DB search query and the web search query.

    Examples:
        "vegetarian", "vegan", "gluten-free", "none"
    """

    dietary_preferences: str
