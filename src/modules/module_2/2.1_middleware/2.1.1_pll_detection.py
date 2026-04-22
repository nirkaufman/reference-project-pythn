from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware


model = init_chat_model(model='gpt-5-nano',
                        temperature=1,
                        max_tokens=1000,
                        timeout=6000)

booking_assistant_prompt = """
    You are a booking assistant.
    Your task is to help users book flights.
    You can have a conversation with the user on any topic
    related to booking flights, and you can ask questions about
    the user's preferences.
    If pll is detected, format the tool output into a readable string 
    instead of structured format
    always return an answer.
""".strip()

pll_detection = create_agent(
    model,
    system_prompt=booking_assistant_prompt,
    middleware=[
        PIIMiddleware("email", strategy="mask", apply_to_input=True),
        PIIMiddleware("url", strategy="block", apply_to_input=True),
    ],
)
