# Re-export the agent so langgraph.json can reference this package directly.
from .agent import agent

__all__ = ["agent"]
