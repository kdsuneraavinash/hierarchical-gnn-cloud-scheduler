from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAgent(nn.Module, ABC):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            value: torch.Tensor = self.get_value_unbatched(x[batch_index])
            values.append(value)

        return torch.stack(values).to(self.device)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        batch_size = x.shape[0]
        all_chosen_actions, all_log_probs, all_entropies, all_values = [], [], [], []

        for batch_index in range(batch_size):
            x_i = x[batch_index]
            action_i = action[batch_index] if action is not None else None
            chosen_action, total_log_prob, total_entropy, value = self.get_action_and_value_unbatched(x_i, action_i)
            all_chosen_actions.append(chosen_action)
            all_log_probs.append(total_log_prob)
            all_entropies.append(total_entropy)
            all_values.append(value)

        chosen_actions = torch.stack(all_chosen_actions).to(self.device)
        log_probs = torch.stack(all_log_probs).to(self.device)
        entropies = torch.stack(all_entropies).to(self.device)
        values = torch.stack(all_values).to(self.device)

        return chosen_actions, log_probs, entropies, values

    @abstractmethod
    def get_value_unbatched(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_action_and_value_unbatched(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
