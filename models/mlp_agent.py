import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from constants import NUM_TASK_FEATURES
from models.gin_agent import GinAgent as BaseAgent
from models.gin_agent import GinAgentActor as BaseAgentActor
from models.gin_agent import GinAgentCritic as BaseAgentCritic
from models.gin_agent import GinTaskEncoder as BaseTaskEncoder


class MlpTaskEncoder(BaseTaskEncoder):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__(hidden_dim, embedding_dim, device)

        # [Nt] -> [H] -> [H] -> [E]
        self.network = nn.Sequential(
            nn.Linear(NUM_TASK_FEATURES, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

    def forward(self, task_features: torch.Tensor, dependencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_encoding: torch.Tensor = self.network(task_features)  # (Nt, E)
        task_pool = global_mean_pool(task_encoding, batch=None)  # (1, E)

        return task_encoding, task_pool


# Actor and Critic
# ------------------------------------------------------------------------------------------------------------------


class MlpAgentActor(BaseAgentActor):
    def __init__(self, device: torch.device, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.task_encoder = MlpTaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)


class MlpAgentCritic(BaseAgentCritic):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.task_encoder = MlpTaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)


# Mlp Agent
# ------------------------------------------------------------------------------------------------------------------


class MlpAgent(BaseAgent):
    def __init__(self, device: torch.device, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.actor = MlpAgentActor(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.critic = MlpAgentCritic(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
