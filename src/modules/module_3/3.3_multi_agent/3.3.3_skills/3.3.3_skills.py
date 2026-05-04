"""Skills architecture demo: A coding assistant with on-demand specializations.

SKILLS vs HANDOFFS vs SUPERVISOR
---------------------------------
Supervisor (3.3.1): A central agent routes tasks to separate sub-agents as tools.
  Many agents, one orchestrator. Best for: parallel, independent domain delegation.

Handoffs (3.3.2): One agent that transitions between roles via state.
  One agent, sequential steps enforced by state. Best for: multi-stage workflows.

Skills (this file): One agent that loads specialized prompts on-demand.
  One agent, many possible specializations, no enforced order. The agent decides
  WHICH skill to activate based on the user's intent — not a pre-defined flow.
  Best for: open-ended assistants where the domain isn't known upfront.

HOW THE SKILLS PATTERN WORKS
------------------------------
The agent starts with a lightweight base prompt listing available skills.
When the user's request requires specialization, the agent calls `load_skill`
with the skill name. The tool returns a rich, focused prompt (the skill's
domain knowledge, conventions, and instructions) which the agent injects into
its reasoning for that turn.

This is "progressive disclosure" — domain knowledge is loaded only when needed,
keeping the base context window small and focused.

Skills in this demo:
  - write_sql    : SQL query expert (SELECT, JOINs, aggregations, CTEs)
  - review_code  : Code reviewer (bugs, style, security, performance)
  - write_tests  : Test author (pytest, edge cases, test structure)
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Skill registry — each skill is a focused prompt defining a specialization.
# In a real system these could be loaded from files, a database, or an API.
# ---------------------------------------------------------------------------

_SKILLS: dict[str, str] = {
    "write_sql": """
        You are an expert SQL query author.
        Guidelines:
        - Prefer CTEs over nested subqueries for readability.
        - Always alias tables and columns clearly.
        - Add brief inline comments for non-obvious logic.
        - Use ANSI SQL unless the user specifies a dialect (PostgreSQL, MySQL, etc.).
        - Point out missing indexes or performance pitfalls when relevant.
        Always return the final SQL in a fenced ```sql block.
    """.strip(),

    "review_code": """
        You are an expert code reviewer focused on correctness, clarity, and security.
        When reviewing code:
        1. Identify bugs or logic errors first.
        2. Flag security issues (injection, secrets in code, unsafe deserialization).
        3. Note performance concerns (N+1 queries, unnecessary allocations).
        4. Suggest readability improvements (naming, structure, comments).
        5. Praise what is done well — reviews should be constructive.
        Structure your response as: Summary → Issues → Suggestions → Verdict.
    """.strip(),

    "write_tests": """
        You are an expert test author using pytest.
        When writing tests:
        - Cover the happy path, edge cases, and error paths.
        - Use descriptive test names: test_<function>_<scenario>_<expected>.
        - Prefer fixtures over setUp/tearDown.
        - Mock only external I/O (network, disk, time) — never internal logic.
        - Add a one-line docstring to each test explaining what it verifies.
        Always return complete, runnable test files.
    """.strip(),
}


# ---------------------------------------------------------------------------
# load_skill tool — the single entry point for progressive disclosure.
# The agent calls this when it determines a skill is needed; the returned
# prompt is injected into the agent's context for that turn.
# ---------------------------------------------------------------------------

@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized skill prompt on-demand.

    Available skills:
    - write_sql   : Expert SQL query author
    - review_code : Expert code reviewer (bugs, security, performance)
    - write_tests : Expert pytest test author

    Returns the skill's full prompt and domain guidelines.
    """
    skill = _SKILLS.get(skill_name)
    if not skill:
        available = ", ".join(_SKILLS)
        return f"Unknown skill '{skill_name}'. Available: {available}"
    return skill


# ---------------------------------------------------------------------------
# Agent — starts lightweight; skills expand its knowledge on-demand.
# ---------------------------------------------------------------------------

model = init_chat_model(model="gpt-4o")

skills_agent = create_agent(
    model,
    tools=[load_skill],
    system_prompt=(
        "You are a coding assistant with access to specialized skills. "
        "Available skills: write_sql, review_code, write_tests. "
        "When the user's request falls within a skill's domain, call load_skill "
        "first to load the relevant expertise, then complete the task using those guidelines. "
        "For general questions, answer directly without loading a skill."
    ),
)
