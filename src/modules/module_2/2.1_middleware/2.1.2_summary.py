from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

model = init_chat_model(model='gpt-5')
summarization_model = init_chat_model(model='gpt-5-nano')

personal_assistant_prompt = """
    You are a personal assistant that like to chat.
    your name is: "Amanda".
    your expertise is: "brainstorming".
""".strip()

summarization = create_agent(
    model,
    system_prompt=personal_assistant_prompt,
    middleware=[
        SummarizationMiddleware(
            model=summarization_model,
            trigger=("tokens", 200),
            keep=("messages", 2),
        ),
    ],
)
