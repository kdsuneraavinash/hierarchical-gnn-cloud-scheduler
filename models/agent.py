import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.functional import softmax
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.nn.models import GIN

from env.observation import EnvObsTensor, unmap_env_obs

# Base Gin Network (Batched Version)
# ------------------------------------------------------------------------------------------------------------------


class BaseGinNetwork(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.task_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)
        self.vm_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

        self.graph_network = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
        ).to(device)

    def __call__(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: EnvObsTensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tasks = obs.task_state_scheduled.shape[0]
        num_vms = obs.vm_completion_time.shape[0]

        # --- Encode Tasks ---
        # ==stack=> [Nt, 3] ==MLP=> [Nt, E]
        task_features = torch.stack([obs.task_state_scheduled, obs.task_state_ready, obs.task_length], dim=-1)
        task_h: torch.Tensor = self.task_encoder(task_features)

        # --- Encode VMs ---
        # ==stack=> [Nv, 3] ==MLP=> [Nv, E]
        vm_features = torch.stack([obs.vm_completion_time, 1.0 / (obs.vm_speed + 1e-8), obs.vm_energy_rate], dim=-1)
        vm_h: torch.Tensor = self.vm_encoder(vm_features)

        # --- Build Graphs ---
        # [Nt, E] & [Nv, E] ==concat=> [(Nt+Nv), E]
        node_x = torch.cat([task_h, vm_h])
        # [Ntt, E] & [Ntv, E] ==concat=> [(Ntv+Ntt), E]
        # Structuring nodes as [0, 1, ..., T-1] [T, T+1, ..., T+VM-1]
        # Edges are between Tasks -> Compatible VMs, and Task -> Task dependencies.
        task_vm_edges = obs.compatibilities.clone()
        task_vm_edges[1] = task_vm_edges[1] + num_tasks  # Reindex VMs
        task_task_edges = obs.task_dependencies
        edge_index = torch.cat([task_vm_edges, task_task_edges], dim=-1)

        # --- Apply GIN ---
        # ==node_embeddings=> [(Nt+Nv), E]
        # ==edge_embeddings=> [(Ntv+Ntt), 2*E]
        # ==graph_embedding=> [1, E]
        batch_vector = torch.zeros(num_tasks + num_vms, dtype=torch.long, device=self.device)
        node_embeddings: torch.Tensor = self.graph_network(node_x, edge_index=edge_index)
        edge_embeddings = torch.cat([node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]], dim=1)
        graph_embedding: torch.Tensor = global_mean_pool(node_embeddings, batch=batch_vector)
        return node_embeddings, edge_embeddings, graph_embedding


# Gin Actor
# ------------------------------------------------------------------------------------------------------------------


class GinActor(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.network = BaseGinNetwork(hidden_dim, embedding_dim, device)
        self.edge_scorer = nn.Sequential(
            nn.Linear(3 * embedding_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: EnvObsTensor) -> torch.Tensor:
        num_tasks = obs.task_state_scheduled.shape[0]
        num_vms = obs.vm_completion_time.shape[0]

        # Run the base network.
        _, edge_embeddings, graph_embedding = self.network(obs)

        # Repeat graph embedding to match edge_embeddings.
        # [1, E] ==repeat=> [(Ntv+Ntt), E] ==concat=> [(Ntv+Ntt), 3*E] ==MLP=> [(Ntv+Ntt), 1] ==flatten=> [(Ntv+Ntt)]
        rep_graph_embedding = graph_embedding.repeat_interleave(edge_embeddings.shape[0], dim=0)
        edge_emb_combined = torch.cat([edge_embeddings, rep_graph_embedding], dim=1)
        edge_scores: torch.Tensor = self.edge_scorer(edge_emb_combined)
        edge_scores = edge_scores.flatten()

        # Extract only the scores corresponding to the compatibilities edges.
        # [(Ntv+Ntt)] ==slice=> [Ntv]
        num_task_vm_edges = obs.compatibilities.shape[-1]
        task_vm_edge_scores = edge_scores[:num_task_vm_edges]

        # Actions scores should be the value in edge embedding, but -inf on invalid actions
        # ==action_scores=> [Nt, Nv]
        action_scores = torch.full((num_tasks, num_vms), -1e8, device=self.device)
        action_scores[obs.compatibilities[0], obs.compatibilities[1]] = task_vm_edge_scores
        action_scores[obs.task_state_ready == 0, :] = -1e8
        return action_scores


# Gin Critic
# ------------------------------------------------------------------------------------------------------------------


class GinCritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.network = BaseGinNetwork(hidden_dim, embedding_dim, device)
        self.graph_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: EnvObsTensor) -> torch.Tensor:
        # Critic value is derived from global graph state
        _, _, graph_embedding = self.network(obs)
        return self.graph_scorer(graph_embedding.unsqueeze(0))


# Gin Agent
# ------------------------------------------------------------------------------------------------------------------


class GinAgent(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        self.actor = GinActor(hidden_dim=64, embedding_dim=16, device=device)
        self.critic = GinCritic(hidden_dim=64, embedding_dim=16, device=device)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            decoded_obs = unmap_env_obs(x[batch_index])
            value = self.critic(decoded_obs)
            values.append(value)

        return torch.stack(values).to(self.device)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        batch_size = x.shape[0]
        all_chosen_actions, all_log_probs, all_entropies, all_values = [], [], [], []

        for batch_index in range(batch_size):
            decoded_obs = unmap_env_obs(x[batch_index])
            action_scores = self.actor(decoded_obs)
            action_scores = action_scores.flatten()
            action_probabilities = softmax(action_scores, dim=0)

            probs = Categorical(action_probabilities)
            chosen_action = action[batch_index] if action is not None else probs.sample()
            value = self.critic(decoded_obs)

            all_chosen_actions.append(chosen_action)
            all_log_probs.append(probs.log_prob(chosen_action))
            all_entropies.append(probs.entropy())
            all_values.append(value)

        chosen_actions = torch.stack(all_chosen_actions).to(self.device)
        log_probs = torch.stack(all_log_probs).to(self.device)
        entropies = torch.stack(all_entropies).to(self.device)
        values = torch.stack(all_values).to(self.device)

        return chosen_actions, log_probs, entropies, values
