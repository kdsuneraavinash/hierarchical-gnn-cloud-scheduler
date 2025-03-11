from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAgent(nn.Module, ABC):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    @abstractmethod
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def get_action_unbatched(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
