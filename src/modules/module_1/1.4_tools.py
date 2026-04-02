from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from pydantic import BaseModel
from langchain.tools import tool

from tavily import TavilyClient

# TavilyClient reads TAVILY_API_KEY from the environment automatically
tavily_client = TavilyClient()

model= init_chat_model(model='gpt-5-nano')

personal_chef_prompt = """
    You are a personal chef assistant. 
    Your task is to provide personalized meal recommendations based on user 
    Ingredients and preferences.
    You can use the web to search for recipes. 
    Generate only one recipe.
"""

# simple arguments for a custom tool
@tool('recipes_web_search')
def search_recipes(ingredients: str) -> str:
    """Search for recipes on the internet based on the given ingredients."""
    return f"No results found for {ingredients}"

@tool('recipes_web_search')
def search_recipes_tavily(ingredients: str) -> dict:
    """Search for recipes on the internet based on the given ingredients."""
    return tavily_client.search(f" recipe with {ingredients}")

# The output schema for the agent
class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    instructions: str

# Putting it all together
tools = create_agent(model=model,
                     tools=[search_recipes_tavily],
                     system_prompt=personal_chef_prompt,
                     response_format=Recipe)
