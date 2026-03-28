from langchain.tools import ToolRuntime, tool
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model

model= init_chat_model(model='gpt-4o')

personal_chef_prompt = """
    You are a personal chef assistant. 
    By accepting an image from the user you can: describe the image, identify the food in the image, and provide a recipe.
"""

# Putting it all together
multimodal = create_agent(
    model=model,
    system_prompt=personal_chef_prompt,
)
