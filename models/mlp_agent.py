import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from constants import NUM_TASK_FEATURES
from models.gin_agent import GinAgent, GinTaskEncoder


class MlpTaskEncoder(GinTaskEncoder):
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


class MlpAgent(GinAgent):
    def __init__(self, device: torch.device, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__(device, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.task_encoder = MlpTaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
