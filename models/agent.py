import torch

from models.base_agent import BaseAgent
from models.gin_agent import GinAgent


def make_agent(agent_type: str | None, device: torch.device) -> BaseAgent:
    if agent_type == "gin":
        return GinAgent(device=device)
    raise ValueError("Agent type is not known")
