import torch
import torch.nn as nn
from torch_geometric.nn.models import GAT

from constants import F_TASK, F_VM, N_TASK, N_VM
from env.observation import decode_env_obs_batched
from models.base_agent import BaseAgent

# Encoders
# ------------------------------------------------------------------------------------------------------------------


class GnnTaskEncoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.network = GAT(  # [Nt] -> [H] -> [H] -> [H] -> [E]
            in_channels=F_TASK,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
            heads=4,
            concat=False,
        ).to(device)

    def forward(self, task_features: torch.Tensor, dependencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = task_features.shape[0]

        task_features_flat = task_features.reshape(B * N_TASK, F_TASK)  # (B*Nt, Ft)
        edge_indices: list[torch.Tensor] = []  # (B, 2, variable)
        for i in range(B):
            edge_index = torch.nonzero(dependencies[i], as_tuple=False).T  # (2, variable)
            edge_indices.append(edge_index + N_TASK * i)

        edge_indices_flat = torch.cat(edge_indices, dim=1)  # (2, Nx)
        task_encoding_flat: torch.Tensor = self.network(task_features_flat, edge_index=edge_indices_flat)  # (B*Nt, E)
        task_encodings = task_encoding_flat.reshape(B, N_TASK, self.embedding_dim)  # (B, Nt, E)
        task_pool = task_encodings.mean(dim=1)  # (B, E)

        return task_encodings, task_pool


class GnnVmEncoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(  # [Nv] -> [H] -> [H] -> [E]
            nn.Linear(F_VM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

    def forward(self, vm_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = vm_features.shape[0]
        vm_features_flat = vm_features.reshape(B * N_VM, F_VM)  # (B*Nv, Fv)
        vm_encoding_flat: torch.Tensor = self.network(vm_features_flat)  # (B*Nv, E)
        vm_encoding = vm_encoding_flat.reshape(B, N_VM, self.embedding_dim)  # (B, Nv, E)
        vm_pool = vm_encoding.mean(dim=1)  # (B, E)

        return vm_encoding, vm_pool


# Decoders
# ------------------------------------------------------------------------------------------------------------------


class GnnDecoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.network = nn.Sequential(  # [3E] -> [H] -> [H] -> [1]
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(
        self, encoding: torch.Tensor, mask: torch.Tensor, task_pool: torch.Tensor, vm_pool: torch.Tensor
    ) -> torch.Tensor:
        B = encoding.shape[0]
        num_x = encoding.shape[1]
        encoding_flat = encoding.reshape(B * num_x, -1)  # (B*Nx, E)
        rep_task_pool = task_pool.repeat(num_x, 1)  # (B*Nx, E)
        rep_vm_pool = vm_pool.repeat(num_x, 1)  # (B*Nx, E)
        comb_encoding = torch.cat([encoding_flat, rep_task_pool, rep_vm_pool], dim=1)  # (B*Nx, 3E)
        scores_flat: torch.Tensor = self.network(comb_encoding)  # (B*Nx, 1)
        scores = scores_flat.reshape(B, num_x)  # (B, Nx)

        scores.masked_fill_(mask == 0, -1e8)
        return scores


# Agent Actor
# ------------------------------------------------------------------------------------------------------------------


class GnnAgentActor(nn.Module):
    def __init__(self, device: torch.device, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.device = device
        self.task_encoder = GnnTaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.vm_encoder = GnnVmEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.task_decoder = GnnDecoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.vm_decoder = GnnDecoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)

    def forward(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded_obs = decode_env_obs_batched(x.to(self.device))
        B = decoded_obs.vm_features.shape[0]

        # --- Encode Tasks ---
        task_features = decoded_obs.task_features  # (B, Nt, Ft)
        vm_features = decoded_obs.vm_features  # (B, Nt, Nv, Fv)
        dependencies = decoded_obs.task_dependencies  # (B, Nt, Nt)
        task_encoding, task_pool = self.task_encoder(task_features, dependencies)  # (B, Nt, E), (B, E)

        # --- Encode VMs (avg properties across tasks) ---
        avg_vm_features = vm_features.mean(dim=1)  # (B, Nv, Fv)
        _, avg_vm_pool = self.vm_encoder(avg_vm_features)  # (B, E)

        # --- Task Selection ---
        task_mask = decoded_obs.task_mask  # (B, Nt)
        task_logits: torch.Tensor = self.task_decoder(task_encoding, task_mask, task_pool, avg_vm_pool)  # (B, Nt)
        task_probs = torch.softmax(task_logits, dim=1)  # (B, Nt)
        task_dist = torch.distributions.Categorical(task_probs)
        chosen_task = task_dist.sample() if action is None else action // N_VM  # (B,)
        task_log_prob = task_dist.log_prob(chosen_task)  # (B,)
        task_entropy = task_dist.entropy()  # (B,)

        # --- Encode the specific VM (with specific properties to the selected task) ---
        chosen_vm_features = vm_features[torch.arange(B), chosen_task]  # (B, Nv, Fv)
        vm_encoding, vm_pool = self.vm_encoder(chosen_vm_features)  # (B, Nv, E), (B, E)

        # --- VM Selection ---
        chosen_vm_mask = decoded_obs.vm_mask[torch.arange(B), chosen_task]  # (B, Nv)
        vm_logits: torch.Tensor = self.vm_decoder(vm_encoding, chosen_vm_mask, task_pool, vm_pool)  # (B, Nv)
        vm_probs = torch.softmax(vm_logits, dim=1)  # (B, Nv)
        vm_dist = torch.distributions.Categorical(vm_probs)
        chosen_vm = vm_dist.sample() if action is None else action % N_VM  # (B,)
        vm_log_prob = vm_dist.log_prob(chosen_vm)  # (B,)
        vm_entropy = vm_dist.entropy()  # (B,)

        # --- Compute Final Action ---
        chosen_action = chosen_task * N_VM + chosen_vm  # Encode action
        total_log_prob = task_log_prob + vm_log_prob  # Combined log probability
        total_entropy = task_entropy + vm_entropy  # Combined entropy

        return chosen_action, total_log_prob, total_entropy


# Critic Agent
# ------------------------------------------------------------------------------------------------------------------


class GnnAgentCritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.task_encoder = GnnTaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.vm_encoder = GnnVmEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.network = nn.Sequential(  # [2E] -> [H] -> [H] -> [1]
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded_obs = decode_env_obs_batched(x.to(self.device))

        task_features = decoded_obs.task_features  # (B, Nt, Ft)
        dependencies = decoded_obs.task_dependencies  # (B, Nt, Nt)
        _, task_pool = self.task_encoder(task_features, dependencies)  # (B, E)
        avg_vm_features = decoded_obs.vm_features.mean(dim=1)  # (B, Nv, Fv)
        _, avg_vm_pool = self.vm_encoder(avg_vm_features)  # (B, E)

        comb_encoding = torch.cat([task_pool, avg_vm_pool], dim=1)  # (B, 2E)
        state_value: torch.Tensor = self.network(comb_encoding)  # (B, 1)
        return state_value.squeeze(dim=-1)


# GNN Agent
# ------------------------------------------------------------------------------------------------------------------


class GnnAgent(BaseAgent):
    def __init__(self, device: torch.device, embedding_dim: int = 8, hidden_dim: int = 64):
        super().__init__(device)
        self.device = device

        self.actor = GnnAgentActor(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.critic = GnnAgentCritic(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor = self.critic(x)
        return value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_action, log_prob, entropy = self.actor(x, action)
        value: torch.Tensor = self.critic(x)
        return chosen_action, log_prob, entropy, value
