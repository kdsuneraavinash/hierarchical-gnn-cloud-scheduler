import random

from algorithms.base_greedy import BaseGreedyScheduler
from dataset.models import Dataset
from env.state import SimulationState


class RandomScheduler(BaseGreedyScheduler):
    def __init__(self) -> None:
        super().__init__("Random")

    def select_task(self, dataset: Dataset, state: SimulationState) -> int:
        """Choose the next task (with no preference)."""
        return random.choice([task.id for task in dataset.tasks if state.task_states[task.id].is_ready])

    def select_vm(self, task_id: int, dataset: Dataset, state: SimulationState) -> int:
        """Schedule the task on the next VM in the list."""
        compatible_vms = [vm for vm in dataset.vms if vm.is_compatible(dataset.tasks[task_id], state.vm_states[vm.id])]
        return random.choice([vm.id for vm in compatible_vms])
