from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model('gpt-4o')

agent = create_agent(model, tools=[])
