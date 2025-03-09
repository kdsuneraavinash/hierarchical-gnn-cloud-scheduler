from abc import ABC, abstractmethod

from algorithms.base_dynamic import BaseDynamicScheduler
from dataset.models import Dataset
from env.simulation import Simulation
from env.state import SimulationState


class BaseGreedyScheduler(BaseDynamicScheduler, ABC):
    _simulation: Simulation

    def __init__(self, name: str):
        super().__init__(name)

    def select_task_and_vm(self, dataset: Dataset, state: SimulationState) -> tuple[int, int]:
        next_task_id = self.select_task(dataset, state)
        selected_vm_id = self.select_vm(next_task_id, dataset, state)
        return next_task_id, selected_vm_id

    @abstractmethod
    def select_task(self, dataset: Dataset, state: SimulationState) -> int:
        """Out of the ready tasks, choose the next task to schedule."""
        raise NotImplementedError()

    @abstractmethod
    def select_vm(self, task_id: int, dataset: Dataset, state: SimulationState) -> int:
        """Assign the task to a VM."""
        raise NotImplementedError()
