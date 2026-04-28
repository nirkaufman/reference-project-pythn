from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

@tool(description="Write a file to disk.")
def write_file(filename: str, content: str) -> str:
    with open(filename, "w") as f:
        f.write(content)
    return f"Wrote {filename} to disk."

@tool(description="Execute an SQL query.")
def execute_sql(query: str) -> str:
    return f"Executing SQL query: {query}"


@tool(description="Read data from a database.")
def read_data(query: str) -> str:
    return f"Reading data from database: {query}"

model = init_chat_model(model='gpt-5-nano')

# noinspection PyTypeChecker
human_loop = create_agent(
    model=model,
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                "read_data": False,
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    # Human-in-the-loop requires checkpointing to handle interrupts.
    # In production, use a persistent checkpointer like AsyncPostgresSaver.
    # checkpointer=InMemorySaver(),
)