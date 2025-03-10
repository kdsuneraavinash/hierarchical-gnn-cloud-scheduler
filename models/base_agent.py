from abc import ABC

import torch
import torch.nn as nn

from env.observation import decode_env_obs


class BaseAgent(nn.Module, ABC):
    task_actor: nn.Module
    vm_actor: nn.Module
    critic: nn.Module

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            decoded_obs = decode_env_obs(x[batch_index])
            value: torch.Tensor = self.critic(decoded_obs)
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

            # --- Task Selection ---
            task_response: tuple[torch.Tensor, torch.Tensor] = self.task_actor(decoded_obs)
            task_logits, task_hs = task_response  # (Nt,), (Nt, E)
            task_probs = torch.softmax(task_logits, dim=0)
            task_dist = torch.distributions.Categorical(task_probs)
            chosen_task = task_dist.sample() if action is None else action[batch_index] // num_vms
            task_log_prob = task_dist.log_prob(chosen_task)
            task_entropy = task_dist.entropy()

            # --- VM Selection ---
            vm_logits: torch.Tensor = self.vm_actor(decoded_obs, task_hs[chosen_task])  # (Nv,)
            vm_probs = torch.softmax(vm_logits, dim=0)
            vm_dist = torch.distributions.Categorical(vm_probs)
            chosen_vm = vm_dist.sample() if action is None else action[batch_index] % num_vms
            vm_log_prob = vm_dist.log_prob(chosen_vm)
            vm_entropy = vm_dist.entropy()

            # --- Compute Final Action & Value ---
            chosen_action = chosen_task * num_vms + chosen_vm  # Encode action
            total_log_prob = task_log_prob + vm_log_prob  # Combined log probability
            total_entropy = task_entropy + vm_entropy  # Combined entropy

            value: torch.Tensor = self.critic(decoded_obs)  # Value estimate from the critic

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

        # --- Task Selection ---
        task_response: tuple[torch.Tensor, torch.Tensor] = self.task_actor(decoded_obs)  # (Nt,)
        task_logits, task_hs = task_response  # (Nt,), (Nt, E)
        task_probs = torch.softmax(task_logits, dim=0)
        task_dist = torch.distributions.Categorical(task_probs)
        chosen_task = task_dist.sample()

        # --- VM Selection ---
        vm_logits: torch.Tensor = self.vm_actor(decoded_obs, task_hs[chosen_task])  # (Nv,)
        vm_probs = torch.softmax(vm_logits, dim=0)
        vm_dist = torch.distributions.Categorical(vm_probs)
        chosen_vm = vm_dist.sample()

        chosen_action: torch.Tensor = chosen_task * num_vms + chosen_vm
        return chosen_action
