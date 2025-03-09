import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from env.observation import EnvObsTensor
from models.base_agent import BaseAgent


def mean_pool(embedding: torch.Tensor, device: torch.device, num_batches: int = 1) -> torch.Tensor:
    batch_vector = torch.arange(num_batches, dtype=torch.long, device=device)
    batch_vector = batch_vector.repeat_interleave(embedding.shape[0] // num_batches)
    mean_pool: torch.Tensor = global_mean_pool(embedding, batch=batch_vector)
    return mean_pool


# Task Agent Actor
# ------------------------------------------------------------------------------------------------------------------


class TaskAgentActor(nn.Module):
    def __init__(self, hidden_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # [2] -> [hidden] -> [hidden] -> [1]
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, obs: EnvObsTensor) -> torch.Tensor:
        task_features = torch.stack([obs.task_completion_time, obs.task_state_scheduled], dim=-1)  # (Nt, 2)
        task_scores: torch.Tensor = self.network(task_features)  # (Nt, 1)
        task_scores = task_scores.flatten()  # (Nt)

        task_scores[obs.task_state_ready == 0] = -1e8
        return task_scores


# VM Agent Actor
# ------------------------------------------------------------------------------------------------------------------


class VmAgentActor(nn.Module):
    def __init__(self, hidden_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # [2] -> [hidden] -> [hidden] -> [embedding]
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, obs: EnvObsTensor, task_id: torch.Tensor) -> torch.Tensor:
        vm_features = torch.stack([obs.vm_completion_time, obs.task_vm_time_cost[task_id]], dim=-1)  # (Nv, 2)
        vm_scores: torch.Tensor = self.network(vm_features)  # (Nv, 1)
        vm_scores = vm_scores.flatten()  # (Nv)

        vm_scores[torch.where(obs.task_vm_compatibilities[task_id] == 0)] = -1e8
        return vm_scores


# Agent Critic
# ------------------------------------------------------------------------------------------------------------------


class AgentCritic(nn.Module):
    def __init__(self, hidden_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # [2] -> [hidden] -> [hidden]
        self.task_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        # [2] -> [hidden] -> [hidden]
        self.vm_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        # [2*hidden_dim] -> [hidden] -> [1]
        self.state_value_network = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, obs: EnvObsTensor) -> torch.Tensor:
        task_features = torch.stack([obs.task_completion_time, obs.task_state_scheduled], dim=-1)  # (Nt, 2)
        task_h: torch.Tensor = self.task_encoder(task_features)  # (Nt, E)
        task_emb = mean_pool(task_h, self.device)  # hG - (1, E)

        vm_features = torch.stack([obs.vm_completion_time, obs.task_vm_time_cost.mean(dim=0)], dim=-1)  # (Nv, 2)
        vm_h: torch.Tensor = self.vm_encoder(vm_features)  # (Nv, E)
        vm_emb = mean_pool(vm_h, self.device)  # (1, E)

        global_emb = torch.cat([task_emb, vm_emb], dim=-1)  # (1, 2E)
        state_value: torch.Tensor = self.state_value_network(global_emb)
        return state_value.squeeze()


# Gin Agent
# ------------------------------------------------------------------------------------------------------------------


class DrlAgent(BaseAgent):
    def __init__(self, device: torch.device):
        super().__init__(device)

        self.task_actor = TaskAgentActor(hidden_dim=64, device=device)
        self.vm_actor = VmAgentActor(hidden_dim=64, device=device)
        self.critic = AgentCritic(hidden_dim=64, device=device)
