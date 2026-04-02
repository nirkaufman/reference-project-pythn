from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model= init_chat_model(model='gpt-5-nano')

personal_chef_prompt = """
    You are a personal chef assistant. 
    Your task is to provide personalized meal recommendations based on user 
    Ingredients and preferences. 
"""

prompt = create_agent(model=model, system_prompt=personal_chef_prompt)
