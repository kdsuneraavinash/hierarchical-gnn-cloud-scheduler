import random

from algorithms.base_greedy import BaseGreedyScheduler
from dataset.models import Dataset
from env.state import SimulationState


class LeastLoadedFirstScheduler(BaseGreedyScheduler):
    """
    Assigns tasks to the least loaded VM at that moment.

    Task Selection: Random from ready tasks,
    VM Selection: The least loaded VM (the one with the earliest completion time)
    """

    def __init__(self, name: str | None = None):
        super().__init__(name or "Least Loaded First")

    def select_task(self, dataset: Dataset, state: SimulationState) -> int:
        """Choose a random task from the ready tasks"""
        ready_tasks = [task.id for task in dataset.tasks if state.task_states[task.id].is_ready]
        return random.choice(ready_tasks)

    def select_vm(self, task_id: int, dataset: Dataset, state: SimulationState) -> int:
        """Assign the task to the least loaded VM (the one with the earliest completion time)."""
        return min(dataset.vms, key=lambda vm: state.vm_states[vm.id].completion_time).id
