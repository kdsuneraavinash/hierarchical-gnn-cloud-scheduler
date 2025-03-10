import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GIN

from constants import NUM_TASK_FEATURES, NUM_VM_FEATURES
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
    def __init__(self, utility_vector: nn.Parameter, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.utility_vector = utility_vector
        # [NUM_TASK_FEATURES] -> [hidden] -> [hidden] -> [hidden] -> [embedding]
        self.task_encoder = GIN(
            in_channels=NUM_TASK_FEATURES,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
        ).to(device)
        # [3*embedding] -> [hidden] -> [1]
        self.task_decoder = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, obs: EnvObsTensor) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Encode tasks ---
        task_features = obs.task_features  # (Nt, Ft)
        task_hs: torch.Tensor = self.task_encoder(task_features, edge_index=obs.task_dependencies)  # hv^L - (Nt, E)
        graph_embedding = mean_pool(task_hs, self.device)  # hG - (1, E)

        # --- Decode tasks ---
        rep_graph_embedding = graph_embedding.repeat(task_hs.shape[0], 1)  # (Nt, E)
        rep_utility_vector = self.utility_vector.unsqueeze(0).repeat(task_hs.shape[0], 1)  # (Nt, E)
        task_embedding = torch.cat([task_hs, rep_graph_embedding, rep_utility_vector], dim=1)  # (Nt, 3E)
        task_scores: torch.Tensor = self.task_decoder(task_embedding)  # (Nt, 1)
        task_scores = task_scores.flatten()  # (Nt)

        task_scores[obs.task_mask == 0] = -1e8
        return task_scores, task_hs


# VM Agent Actor
# ------------------------------------------------------------------------------------------------------------------


class VmAgentActor(nn.Module):
    def __init__(self, utility_vector: nn.Parameter, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.utility_vector = utility_vector
        # [NUM_VM_FEATURES] -> [hidden] -> [embedding]
        self.vm_encoder = nn.Sequential(
            nn.Linear(NUM_VM_FEATURES, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)
        # [4*embedding] -> [hidden] -> [1]
        self.vm_decoder = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, obs: EnvObsTensor, task_h: torch.Tensor) -> torch.Tensor:
        # --- Encode VMs ---
        vm_features = obs.vm_features  # (Nv, Fv)
        vm_hs: torch.Tensor = self.vm_encoder(vm_features)  # hv^L - (Nv, E)
        graph_embedding = mean_pool(vm_hs, self.device)  # hG - (1, E)

        # --- Decode VMs ---
        rep_graph_embedding = graph_embedding.repeat(vm_hs.shape[0], 1)  # (Nv, E)
        rep_utility_vector = self.utility_vector.unsqueeze(0).repeat(vm_hs.shape[0], 1)  # (Nv, E)
        rep_task_h = task_h.unsqueeze(0).repeat(vm_hs.shape[0], 1)  # Nv, E
        vm_embedding = torch.cat([vm_hs, rep_graph_embedding, rep_task_h, rep_utility_vector], dim=1)  # (Nv, 4E)
        vm_scores: torch.Tensor = self.vm_decoder(vm_embedding)  # (Nv, 1)
        vm_scores = vm_scores.flatten()  # (Nv)

        # vm_scores[torch.where(obs.task_vm_compatibilities[task_id] == 0)] = -1e8
        return vm_scores


# Agent Critic
# ------------------------------------------------------------------------------------------------------------------


class AgentCritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # [NUM_TASK_FEATURES] -> [hidden] -> [hidden] -> [hidden] -> [embedding]
        self.task_encoder = GIN(
            in_channels=NUM_TASK_FEATURES,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
        ).to(device)
        # [NUM_VM_FEATURES] -> [hidden] -> [embedding]
        self.vm_encoder = nn.Sequential(
            nn.Linear(NUM_VM_FEATURES, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)
        # [2*embedding] -> [hidden] -> [1]
        self.state_value_network = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, obs: EnvObsTensor) -> torch.Tensor:
        task_features = obs.task_features  # (Nt, Ft)
        task_h: torch.Tensor = self.task_encoder(task_features, edge_index=obs.task_dependencies)  # hv^L - (Nt, E)
        task_graph_embedding = mean_pool(task_h, self.device)  # hG - (1, E)

        vm_features = obs.vm_features  # (Nv, Fv)
        vm_h: torch.Tensor = self.vm_encoder(vm_features)  # hv^L - (Nv, E)
        vm_graph_embedding = mean_pool(vm_h, self.device)  # hG - (1, E)

        global_embedding = torch.cat([task_graph_embedding, vm_graph_embedding], dim=-1)  # (1, 2E)
        state_value: torch.Tensor = self.state_value_network(global_embedding)
        return state_value.squeeze()


# Gin Agent
# ------------------------------------------------------------------------------------------------------------------


class GinAgent(BaseAgent):
    def __init__(self, device: torch.device):
        super().__init__(device)

        embedding_dim = 16
        utility_vector = nn.Parameter(torch.randn(embedding_dim, device=device))
        self.task_actor = TaskAgentActor(utility_vector, hidden_dim=64, embedding_dim=embedding_dim, device=device)
        self.vm_actor = VmAgentActor(utility_vector, hidden_dim=64, embedding_dim=embedding_dim, device=device)
        self.critic = AgentCritic(hidden_dim=64, embedding_dim=embedding_dim, device=device)
