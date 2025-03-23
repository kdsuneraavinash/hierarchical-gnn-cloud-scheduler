import torch
import torch.nn as nn

from constants import F_TASK, N_TASK
from models.base_agent import BaseAgent
from models.gnn_agent import GnnAgentActor, GnnAgentCritic


# Task Encoder
# ------------------------------------------------------------------------------------------------------------------


class MlpTaskEncoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(  # [Nv] -> [H] -> [H] -> [H] -> [E]
            nn.Linear(F_TASK, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

    def forward(self, task_features: torch.Tensor, dependencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = task_features.shape[0]

        task_features_flat = task_features.reshape(B * N_TASK, F_TASK)  # (B*Nt, Ft)
        task_encoding_flat: torch.Tensor = self.network(task_features_flat)  # (B*Nt, E)
        task_encoding = task_encoding_flat.reshape(B, N_TASK, self.embedding_dim)  # (B, Nt, E)
        task_pool = task_encoding.mean(dim=1)  # (B, E)

        return task_encoding, task_pool


# Agent and Critic
# ------------------------------------------------------------------------------------------------------------------


class MlpAgentActor(GnnAgentActor):
    def __init__(self, device: torch.device, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.task_encoder = MlpTaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)


class MlpAgentCritic(GnnAgentCritic):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.task_encoder = MlpTaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)


# MLP Agent
# ------------------------------------------------------------------------------------------------------------------


class MlpAgent(BaseAgent):
    def __init__(self, device: torch.device, embedding_dim: int = 8, hidden_dim: int = 64):
        super().__init__(device)
        self.device = device

        self.actor = MlpAgentActor(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.critic = MlpAgentCritic(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor = self.critic(x)
        return value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_action, log_prob, entropy = self.actor(x, action)
        value: torch.Tensor = self.critic(x)
        return chosen_action, log_prob, entropy, value
