from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from typing import Callable
from langchain_core.tools import tool
from dataclasses import dataclass


@tool(description="Read a file from disk.")
def read_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()

@tool(description="Write to a file.")
def write_file(filename: str, content: str) -> str:
    with open(filename, "w") as f:
        f.write(content)

    return f"File {filename} written successfully."

@dataclass
class Context:
    user_role: str

@wrap_model_call
def context_based_tools(
        request: ModelRequest[Context],
        handler: Callable[[ModelRequest[Context]], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Runtime Context permissions."""
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        pass
    elif user_role == "editor":
        tools = [t for t in request.tools if t.name.startswith("write_")]
        request = request.override(tools=tools)
    else:
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)

    print(f"Selected tools: {request.tools}")
    return handler(request)

model = init_chat_model(model='gpt-5-nano')
all_tools = [read_file, write_file]

runtime_tools = create_agent(model=model,
                             tools=all_tools,
                             middleware=[context_based_tools],
                             context_schema=Context)
