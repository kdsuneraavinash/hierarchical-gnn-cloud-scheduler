from typing import Literal
import torch

from models.base_agent import BaseAgent
from models.gnn_agent import GnnAgent
from models.mlp_agent import MlpAgent


AgentType = Literal["gnn", "mlp"]


def make_agent(agent_type: AgentType | None, device: torch.device) -> BaseAgent:
    if agent_type == "gnn":
        return GnnAgent(device=device)
    if agent_type == "mlp":
        return MlpAgent(device=device)
    raise ValueError("Agent type is not known")
