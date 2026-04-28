from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

client = Client()

# 1. Developer pulls the latest prompt version
# NOTE: Replace "sales-lead-extractor" with an actual prompt from your LangSmith workspace
try:
    pulled_prompt = client.pull_prompt("sales-lead-extractor")
except (RuntimeError, ValueError):
    pulled_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sales lead extraction assistant. Extract key information from call summaries."),
        ("human", "{call_summary}"),
    ])

# 2. In a real app, this summary string comes directly from the 11labs webhook/API response
mock_conversation_summary = """
    Outbound call connected with Elena Rodriguez, VP of Engineering at CloudScale Inc. Elena mentioned they 
    are actively looking for a new cloud infrastructure provider because their current solution is causing severe 
    latency issues during peak hours, and their budget for Q3 is around $15k/month. I proposed a discovery 
    meeting with our senior sales rep, David. Elena agreed to an in-person meeting next Tuesday, May 5th at 2:00 PM. 
    She asked if David could bring a technical whitepaper on our failover protocols. The meeting will be at her office
     located at 890 Tech Boulevard, Building B, Seattle, WA. Call ended positively.
"""

# 3. Format the prompt with the variable
formatted_prompt = pulled_prompt.invoke({
    "call_summary": mock_conversation_summary
})

model = init_chat_model(model='gpt-5-nano')

sales_lead_agent = create_agent(
    model=model,
    system_prompt=formatted_prompt.messages[0].content,
    tools=[],
)
