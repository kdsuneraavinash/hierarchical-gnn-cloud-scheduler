import torch
import torch.nn as nn

from constants import F_TASK, N_TASK
from models.gnn_agent import GnnAgent as BaseAgent
from models.gnn_agent import GnnAgentActor as BaseAgentActor
from models.gnn_agent import GnnAgentCritic as BaseAgentCritic
from models.gnn_agent import GnnTaskEncoder as BaseTaskEncoder


class MlpTaskEncoder(BaseTaskEncoder):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__(hidden_dim, embedding_dim, device)

        # [Nt] -> [H] -> [H] -> [E]
        self.network = nn.Sequential(
            nn.Linear(F_TASK, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

    def forward(self, task_features: torch.Tensor, dependencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = task_features.shape[0]
        task_features_flat = task_features.reshape(B * N_TASK, -1)  # (B*Nt, Ft)
        task_encoding_flat: torch.Tensor = self.network(task_features_flat)  # (B*Nt, E)
        task_encoding = task_encoding_flat.reshape(B, N_TASK, -1)  # (B, Nt, E)
        task_pool = task_encoding.mean(dim=1)  # (B, E)

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
