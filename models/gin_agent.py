import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GIN

from constants import NUM_TASK_FEATURES, NUM_VM_FEATURES
from env.observation import EnvObsTensor, decode_env_obs
from models.base_agent import BaseAgent

# Encoders
# ------------------------------------------------------------------------------------------------------------------


class GinTaskEncoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.network = GIN(  # [Nt] -> [H] -> [H] -> [H] -> [E]
            in_channels=NUM_TASK_FEATURES,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
        ).to(device)

    def forward(self, task_features: torch.Tensor, dependencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_encoding: torch.Tensor = self.network(task_features, edge_index=dependencies)  # (Nt, E)
        task_pool = global_mean_pool(task_encoding, batch=None)  # (1, E)

        return task_encoding, task_pool


class GinVmEncoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.network = nn.Sequential(  # [Nv] -> [H] -> [H] -> [E]
            nn.Linear(NUM_VM_FEATURES, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

    def forward(self, vm_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vm_encoding: torch.Tensor = self.network(vm_features)  # (Nv, E)
        vm_pool = global_mean_pool(vm_encoding, batch=None)  # (1, E)

        return vm_encoding, vm_pool


# Decoders
# ------------------------------------------------------------------------------------------------------------------


class GinDecoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.network = nn.Sequential(  # [3E] -> [H] -> [H] -> [1]
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(
        self, encoding: torch.Tensor, mask: torch.Tensor, task_pool: torch.Tensor, vm_pool: torch.Tensor
    ) -> torch.Tensor:
        rep_task_pool = task_pool.repeat(encoding.shape[0], 1)  # (Nx, E)
        rep_vm_pool = vm_pool.repeat(encoding.shape[0], 1)  # (Nx, E)
        comb_encoding = torch.cat([encoding, rep_task_pool, rep_vm_pool], dim=1)  # (Nx, 3E)
        scores: torch.Tensor = self.network(comb_encoding)  # (Nt, 1)
        scores = scores.flatten()  # (Nt)

        scores[mask == 0] = -1e8
        return scores


# Agent Actor
# ------------------------------------------------------------------------------------------------------------------


class GinAgentActor(nn.Module):
    def __init__(self, device: torch.device, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__(device)
        self.device = device
        self.task_encoder = GinTaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.vm_encoder = GinVmEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.task_decoder = GinDecoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.vm_decoder = GinDecoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)

    def get_action_unbatched(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded_obs = decode_env_obs(x.to(self.device))
        num_vms = decoded_obs.vm_features.shape[1]

        # --- Encode Tasks ---
        task_features = decoded_obs.task_features  # (Nt, Ft)
        dependencies = decoded_obs.task_dependencies
        task_encoding, task_pool = self.task_encoder(task_features, dependencies)  # (Nt, E), (1, E)

        # --- Encode VMs (avg properties across tasks) ---
        avg_vm_features = decoded_obs.vm_features.mean(dim=0)  # (Nv, Fv)
        _, avg_vm_pool = self.vm_encoder(avg_vm_features)  # (Nv, E), (1, E)

        # --- Task Selection ---
        task_mask = decoded_obs.task_mask  # (Nt,)
        task_logits: torch.Tensor = self.task_decoder(task_encoding, task_mask, task_pool, avg_vm_pool)  # (Nt,)
        task_probs = torch.softmax(task_logits, dim=0)
        task_dist = torch.distributions.Categorical(task_probs)
        chosen_task = task_dist.sample() if action is None else action // num_vms
        task_log_prob = task_dist.log_prob(chosen_task)
        task_entropy = task_dist.entropy()

        # --- Encode the specific VM (with task-specific properties) ---
        vm_features = decoded_obs.vm_features[chosen_task]  # (Nv, Fv)
        vm_encoding, vm_pool = self.vm_encoder(vm_features)  # (Nv, E), (1, E)

        # --- VM Selection ---
        vm_mask = decoded_obs.vm_mask  # (Nv,)
        vm_logits: torch.Tensor = self.vm_decoder(vm_encoding, vm_mask, task_pool, vm_pool)  # (Nv,)
        vm_probs = torch.softmax(vm_logits, dim=0)
        vm_dist = torch.distributions.Categorical(vm_probs)
        chosen_vm = vm_dist.sample() if action is None else action % num_vms
        vm_log_prob = vm_dist.log_prob(chosen_vm)
        vm_entropy = vm_dist.entropy()

        # --- Compute Final Action ---
        chosen_action = chosen_task * num_vms + chosen_vm  # Encode action
        total_log_prob = task_log_prob + vm_log_prob  # Combined log probability
        total_entropy = task_entropy + vm_entropy  # Combined entropy

        return chosen_action, total_log_prob, total_entropy


# Critic Agent
# ------------------------------------------------------------------------------------------------------------------


class GinAgentCritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.task_encoder = GinTaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.vm_encoder = GinVmEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.network = nn.Sequential(  # [2E] -> [H] -> [H] -> [1]
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, obs: EnvObsTensor) -> torch.Tensor:
        task_features = obs.task_features  # (Nt, Ft)
        _, task_pool = self.task_encoder(task_features, obs.task_dependencies)  # (1, E)
        avg_vm_features = obs.vm_features.mean(dim=0)  # (Nv, Fv)
        _, avg_vm_pool = self.vm_encoder(avg_vm_features)  # (1, E)

        comb_encoding = torch.cat([task_pool, avg_vm_pool], dim=1)  # (1, 2E)
        state_value: torch.Tensor = self.network(comb_encoding)  # (1, 1)
        return state_value.squeeze()


# Gin Agent
# ------------------------------------------------------------------------------------------------------------------


class GinAgent(BaseAgent):
    def __init__(self, device: torch.device, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__(device)
        self.device = device

        self.actor = GinAgentActor(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.critic = GinAgentCritic(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)

    def get_value_unbatched(self, x: torch.Tensor) -> torch.Tensor:
        decoded_obs = decode_env_obs(x.to(self.device))
        value: torch.Tensor = self.critic(decoded_obs)

        return value

    def get_action_and_value_unbatched(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded_obs = decode_env_obs(x.to(self.device))
        chosen_action, total_log_prob, total_entropy = self.actor(decoded_obs, action)
        value: torch.Tensor = self.critic(decoded_obs)  # Value estimate from the critic

        return chosen_action, total_log_prob, total_entropy, value
