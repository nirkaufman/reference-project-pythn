from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from tavily import TavilyClient


# Sub-agent tools

@tool(description="Search the web for information on a given topic.")
def search_web(query: str) -> str:
    client = TavilyClient()
    results = client.search(query, max_results=3)
    return "\n\n".join(r["content"] for r in results["results"])


@tool(description="Fact-check a claim by searching authoritative sources.")
def fact_check_web(claim: str) -> str:
    client = TavilyClient()
    results = client.search(f"fact check: {claim}", max_results=2)
    return "\n".join(r["content"][:300] for r in results["results"])



# Sub-agents

model = init_chat_model(model="gpt-4o")

_research_agent = create_agent(
    model,
    tools=[search_web],
    system_prompt=(
        "You are a research specialist. Search the web and gather thorough "
        "information on the given topic. Always include your findings in your "
        "final response — the supervisor only sees your last message."
    ),
)

_summary_agent = create_agent(
    model,
    system_prompt=(
        "You are a summarization expert. Condense the provided research into "
        "clear, concise bullet-point key findings. Return only the summary."
    ),
)

_writer_agent = create_agent(
    model,
    system_prompt=(
        "You are a professional report writer. Given a topic and key findings, "
        "produce a well-structured markdown report with an introduction, key "
        "findings section, and a conclusion. Return the full report text."
    ),
)

_fact_checker_agent = create_agent(
    model,
    tools=[fact_check_web],
    system_prompt=(
        "You are a fact-checker. Verify the main claims in the provided report "
        "using authoritative web sources. Return a verdict for each claim "
        "(Verified / Unverified / False) with a brief explanation."
    ),
)

# Sub-agents wrapped as tools for the supervisor

@tool("researcher", description="Research a topic on the web and return detailed findings.")
def call_researcher(query: str) -> str:
    research_agent_result = _research_agent.invoke({"messages": [HumanMessage(content=query)]})
    return research_agent_result["messages"][-1].content


@tool("summarizer", description="Summarize research findings into concise key points.")
def call_summarizer(research: str) -> str:
    summary_agent_result = _summary_agent.invoke({"messages": [HumanMessage(content=research)]})
    return summary_agent_result["messages"][-1].content


@tool("writer", description="Write a structured report given a topic and key findings.")
def call_writer(topic: str, key_findings: str) -> str:
    prompt = f"Topic: {topic}\n\nKey findings:\n{key_findings}"
    writer_agent_result = _writer_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return writer_agent_result["messages"][-1].content


@tool("fact_checker", description="Fact-check the claims in a report and return a verdict.")
def call_fact_checker(report: str) -> str:
    fact_checker_agent_result = _fact_checker_agent.invoke({"messages": [HumanMessage(content=report)]})
    return fact_checker_agent_result["messages"][-1].content

# Supervisor (main agent)

SUPERVISOR_PROMPT = """
    You are a supervisor orchestrating a research and report pipeline.
    You have four specialist sub-agents available as tools:
    
    - researcher    — searches the web and gathers detailed information on a topic
    - summarizer    — condenses raw research into concise key points
    - writer        — writes a structured markdown report from key findings
    - fact_checker  — verifies the accuracy of claims in the final report
    
    Follow this pipeline for every request:
    1. Call researcher to gather information
    2. Call summarizer with the research output
    3. Call writer with the topic and key findings
    4. Call fact_checker with the final report
    5. Return the completed, fact-checked report to the user
""".strip()

supervisor_agent = create_agent(
    model,
    tools=[call_researcher, call_summarizer, call_writer, call_fact_checker],
    system_prompt=SUPERVISOR_PROMPT,
)
