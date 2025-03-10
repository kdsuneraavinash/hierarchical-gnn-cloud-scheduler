import torch

from models.drl_agent import DrlAgent
from models.gat_agent import GatAgent
from models.gin_agent import GinAgent


def make_agent(agent_type: str, device: torch.device):
    if agent_type == "gin":
        return GinAgent(device=device)
    elif agent_type == "gat":
        return GatAgent(device=device)
    elif agent_type == "drl":
        return DrlAgent(device=device)
    raise ValueError("Agent type is not known")
