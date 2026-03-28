from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from pydantic import BaseModel

model= init_chat_model(model='gpt-5-nano')

personal_chef_prompt = """
    You are a personal chef assistant. 
    Your task is to provide personalized meal recommendations based on user 
    Ingredients and preferences. 
    Generate only one recipe.
"""

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    instructions: str

structure = create_agent(model=model,
                      system_prompt=personal_chef_prompt,
                      response_format=Recipe)
