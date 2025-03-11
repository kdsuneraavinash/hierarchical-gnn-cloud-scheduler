import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GIN

from constants import NUM_TASK_FEATURES, NUM_VM_FEATURES
from env.observation import EnvObsTensor, decode_env_obs
from models.base_agent import BaseAgent


def mean_pool(embedding: torch.Tensor, device: torch.device, num_batches: int = 1) -> torch.Tensor:
    batch_vector = torch.arange(num_batches, dtype=torch.long, device=device)
    batch_vector = batch_vector.repeat_interleave(embedding.shape[0] // num_batches)
    mean_pool: torch.Tensor = global_mean_pool(embedding, batch=batch_vector)
    return mean_pool


# Encoders
# ------------------------------------------------------------------------------------------------------------------


class TaskEncoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # [Nt] -> [H] -> [H] -> [H] -> [E]
        self.network = GIN(
            in_channels=NUM_TASK_FEATURES,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
        ).to(device)

    def forward(self, obs: EnvObsTensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_features = obs.task_features  # (Nt, Ft)
        task_encoding: torch.Tensor = self.network(task_features, edge_index=obs.task_dependencies)  # (Nt, E)
        task_pool = mean_pool(task_encoding, self.device)  # (1, E)

        return task_encoding, task_pool


class VmEncoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # [Nv] -> [H] -> [H] -> [E]
        self.network = nn.Sequential(
            nn.Linear(NUM_VM_FEATURES, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

    def forward(self, obs: EnvObsTensor) -> tuple[torch.Tensor, torch.Tensor]:
        vm_features = obs.vm_features  # (Nv, Fv)
        vm_encoding: torch.Tensor = self.network(vm_features)  # (Nv, E)
        vm_pool = mean_pool(vm_encoding, self.device)  # (1, E)

        return vm_encoding, vm_pool


# Agent Actor
# ------------------------------------------------------------------------------------------------------------------


class AgentActor(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # [3E] -> [H] -> [H] -> [1]
        self.network = nn.Sequential(
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


# Critic Actor
# ------------------------------------------------------------------------------------------------------------------


class AgentCritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # [2E] -> [H] -> [H] -> [1]
        self.network = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def forward(self, task_pool: torch.Tensor, vm_pool: torch.Tensor) -> torch.Tensor:
        comb_encoding = torch.cat([task_pool, vm_pool], dim=1)  # (1, 2E)
        state_value: torch.Tensor = self.network(comb_encoding)  # (1, 1)
        return state_value.squeeze()


# Gin Agent
# ------------------------------------------------------------------------------------------------------------------


class GinAgent(BaseAgent):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.device = device

        embedding_dim = 32
        hidden_dim = 64
        self.task_encoder = TaskEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.vm_encoder = VmEncoder(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.task_actor = AgentActor(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.vm_actor = AgentActor(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)
        self.critic = AgentCritic(hidden_dim=hidden_dim, embedding_dim=embedding_dim, device=device)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            decoded_obs = decode_env_obs(x[batch_index])

            task_encoding_response: tuple[torch.Tensor, torch.Tensor] = self.task_encoder(decoded_obs)
            _, task_pool = task_encoding_response  # (Nt, E), (1, E)
            vm_encoding_response: tuple[torch.Tensor, torch.Tensor] = self.vm_encoder(decoded_obs)
            _, vm_pool = vm_encoding_response  # (Nv, E), (1, E)
            value: torch.Tensor = self.critic(task_pool, vm_pool)
            values.append(value)

        return torch.stack(values).to(self.device)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        batch_size = x.shape[0]
        all_chosen_actions, all_log_probs, all_entropies, all_values = [], [], [], []

        for batch_index in range(batch_size):
            decoded_obs = decode_env_obs(x[batch_index])
            num_vms = decoded_obs.vm_features.shape[0]

            # --- Encode ---
            task_encoding_response: tuple[torch.Tensor, torch.Tensor] = self.task_encoder(decoded_obs)
            task_encoding, task_pool = task_encoding_response  # (Nt, E), (1, E)
            vm_encoding_response: tuple[torch.Tensor, torch.Tensor] = self.vm_encoder(decoded_obs)
            vm_encoding, vm_pool = vm_encoding_response  # (Nv, E), (1, E)

            # --- Task Selection ---
            task_mask = decoded_obs.task_mask  # (Nt,)
            task_logits: torch.Tensor = self.task_actor(task_encoding, task_mask, task_pool, vm_pool)  # (Nt,)
            task_probs = torch.softmax(task_logits, dim=0)
            task_dist = torch.distributions.Categorical(task_probs)
            chosen_task = task_dist.sample() if action is None else action[batch_index] // num_vms
            task_log_prob = task_dist.log_prob(chosen_task)
            task_entropy = task_dist.entropy()

            # --- VM Selection ---
            vm_mask = torch.ones(num_vms)  # (Nv,)
            vm_logits: torch.Tensor = self.vm_actor(vm_encoding, vm_mask, task_pool, vm_pool)  # (Nv,)
            vm_probs = torch.softmax(vm_logits, dim=0)
            vm_dist = torch.distributions.Categorical(vm_probs)
            chosen_vm = vm_dist.sample() if action is None else action[batch_index] % num_vms
            vm_log_prob = vm_dist.log_prob(chosen_vm)
            vm_entropy = vm_dist.entropy()

            # --- Compute Final Action & Value ---
            chosen_action = chosen_task * num_vms + chosen_vm  # Encode action
            total_log_prob = task_log_prob + vm_log_prob  # Combined log probability
            total_entropy = task_entropy + vm_entropy  # Combined entropy

            value: torch.Tensor = self.critic(task_pool, vm_pool)  # Value estimate from the critic

            # --- Append Results ---
            all_chosen_actions.append(chosen_action)
            all_log_probs.append(total_log_prob)
            all_entropies.append(total_entropy)
            all_values.append(value)

        chosen_actions = torch.stack(all_chosen_actions).to(self.device)
        log_probs = torch.stack(all_log_probs).to(self.device)
        entropies = torch.stack(all_entropies).to(self.device)
        values = torch.stack(all_values).to(self.device)

        return chosen_actions, log_probs, entropies, values

    def get_action_unbatched(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        decoded_obs = decode_env_obs(x)
        num_vms = decoded_obs.vm_features.shape[0]

        # --- Encode ---
        task_encoding_response: tuple[torch.Tensor, torch.Tensor] = self.task_encoder(decoded_obs)
        task_encoding, task_pool = task_encoding_response  # (Nt, E), (1, E)
        vm_encoding_response: tuple[torch.Tensor, torch.Tensor] = self.vm_encoder(decoded_obs)
        vm_encoding, vm_pool = vm_encoding_response  # (Nv, E), (1, E)

        # --- Task Selection ---
        task_mask = decoded_obs.task_mask  # (Nt,)
        task_logits: torch.Tensor = self.task_actor(task_encoding, task_mask, task_pool, vm_pool)  # (Nt,)
        task_probs = torch.softmax(task_logits, dim=0)
        task_dist = torch.distributions.Categorical(task_probs)
        chosen_task = task_dist.sample()

        # --- VM Selection ---
        vm_mask = torch.ones(num_vms)  # (Nv,)
        vm_logits: torch.Tensor = self.vm_actor(vm_encoding, vm_mask, task_pool, vm_pool)  # (Nv,)
        vm_probs = torch.softmax(vm_logits, dim=0)
        vm_dist = torch.distributions.Categorical(vm_probs)
        chosen_vm = vm_dist.sample()

        chosen_action: torch.Tensor = chosen_task * num_vms + chosen_vm
        return chosen_action
