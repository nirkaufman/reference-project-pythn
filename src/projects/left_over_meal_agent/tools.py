# tools.py
# Defines all tools available to the left-over meal agent.
#
# Concept: Tools
# Tools are Python functions decorated with @tool that the agent can call
# during its reasoning loop. Each tool has a name, a docstring (used by the
# LLM to decide when to call it), and a typed signature.
#
# Concept: Access State (read-only)
# Tools that need user context (dietary preferences) receive a ToolRuntime
# parameter as their first argument. This gives read-only access to the
# agent's current state via runtime.state.get(key, default).
#
# Three tools are defined here:
#   1. search_local_db  — searches a local JSON file for saved recipes
#   2. search_recipes_web — calls Tavily to search the internet for a recipe
#   3. save_recipe_to_db — persists a new recipe to the local JSON file

import json
import os
import sys

# Make sibling files importable when loaded directly by LangGraph
sys.path.insert(0, os.path.dirname(__file__))

from langchain.tools import ToolRuntime, tool
from tavily import TavilyClient

from state import MealAgentState

# Path to the local "database" file — a JSON array of saved recipes.
# Using __file__ makes the path work regardless of where the process is launched.
_DB_PATH = os.path.join(os.path.dirname(__file__), "recipes_db.json")

# TavilyClient reads TAVILY_API_KEY from the environment automatically.
_tavily = TavilyClient()


# ─── Helper: file-based DB read/write ──────────────────────────────────────────

def _load_db() -> list[dict]:
    """Load all saved recipes from the JSON file."""
    with open(_DB_PATH, "r") as f:
        return json.load(f)


def _save_db(recipes: list[dict]) -> None:
    """Write the full recipe list back to the JSON file."""
    with open(_DB_PATH, "w") as f:
        json.dump(recipes, f, indent=2)


def _score_recipe(recipe: dict, query_ingredients: list[str]) -> int:
    """Return how many query ingredients appear in the saved recipe's ingredient list."""
    saved = " ".join(recipe.get("ingredients", [])).lower()
    return sum(1 for ing in query_ingredients if ing.strip().lower() in saved)


# ─── Tool 1: search_local_db ───────────────────────────────────────────────────

@tool
def search_local_db(runtime: ToolRuntime[MealAgentState], ingredients: str) -> str:
    """Search the local recipe database for a recipe that matches the given ingredients.

    Call this tool FIRST before searching the web. It is fast and free.
    Returns the best matching recipe as a JSON string, or a message
    indicating no recipe was found.
    """
    # Read dietary preferences from agent state (read-only access)
    dietary_prefs = runtime.state.get("dietary_preferences", "none")
    print(f"[DB] Searching local database | ingredients: {ingredients} | prefs: {dietary_prefs}")

    recipes = _load_db()
    if not recipes:
        return "No recipes found in the local database."

    # Split ingredient string into individual tokens for scoring
    query_ingredients = [i.strip() for i in ingredients.split(",")]

    # Score every saved recipe and pick the best match
    scored = [(recipe, _score_recipe(recipe, query_ingredients)) for recipe in recipes]
    best_recipe, best_score = max(scored, key=lambda x: x[1])

    # Require at least one matching ingredient to consider it a valid match
    if best_score == 0:
        return "No matching recipe found in the local database."

    print(f"[DB] Found match: '{best_recipe['name']}' (score: {best_score})")
    return json.dumps(best_recipe)


# ─── Tool 2: search_recipes_web ────────────────────────────────────────────────

@tool("web_recipe_search")
def search_recipes_web(runtime: ToolRuntime[MealAgentState], ingredients: str) -> dict:
    """Search the internet for a recipe based on the given ingredients.

    Only call this tool if the local database search returned no results.
    Returns raw search results from Tavily — the agent will extract the
    best recipe from those results.
    """
    # Read dietary preferences to narrow the web search
    dietary_prefs = runtime.state.get("dietary_preferences", "none")

    # Build a focused search query
    prefs_clause = f"{dietary_prefs} " if dietary_prefs and dietary_prefs != "none" else ""
    query = f"{prefs_clause}recipe using {ingredients}"

    print(f"[WEB] Searching Tavily | query: '{query}'")
    return _tavily.search(query)


# ─── Tool 3: save_recipe_to_db ─────────────────────────────────────────────────

@tool
def save_recipe_to_db(name: str, ingredients: str, instructions: str) -> str:
    """Save a newly discovered recipe to the local database for future retrieval.

    Only call this tool after finding a recipe from the web search — do NOT
    call it for recipes that were already retrieved from the local database.

    Args:
        name: The recipe name (e.g. "Spinach and Feta Omelette").
        ingredients: Comma-separated ingredient list.
        instructions: Full cooking instructions as a single string.
    """
    recipes = _load_db()

    # Build the record to store — keep ingredients as a list for easy scoring later
    new_recipe = {
        "name": name,
        "ingredients": [i.strip() for i in ingredients.split(",")],
        "instructions": instructions,
    }

    recipes.append(new_recipe)
    _save_db(recipes)

    print(f"[DB] Saved new recipe: '{name}' ({len(recipes)} total in database)")
    return f"Recipe '{name}' has been saved to the local database."
