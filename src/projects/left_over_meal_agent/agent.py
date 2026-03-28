# NOTE: LangGraph loads this file directly (not as a package), so we insert
# the project directory into sys.path to allow plain imports from sibling
# files (prompt.py, state.py, tools.py) without relative import syntax.

import os
import sys

# Allow importing sibling files as plain modules when loaded by LangGraph
sys.path.insert(0, os.path.dirname(__file__))

from langchain.agents import create_agent  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from prompt import SYSTEM_PROMPT  # noqa: E402
from state import MealAgentState  # noqa: E402
from tools import save_recipe_to_db, search_local_db, search_recipes_web  # noqa: E402


# ─── Structured output schema ──────────────────────────────────────────────────
# Concept: Structured Output
# Pydantic enforces that every response contains these fields before returning.

class Recipe(BaseModel):
    """The final recipe returned to the user."""

    name: str               # A descriptive, appetizing recipe title
    ingredients: list[str]  # Full ingredient list (with quantities if known)
    instructions: str       # Numbered cooking steps as a single string

model = init_chat_model(model="gpt-4o")


agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_local_db, search_recipes_web, save_recipe_to_db],
    response_format=Recipe,
    state_schema=MealAgentState,
)
