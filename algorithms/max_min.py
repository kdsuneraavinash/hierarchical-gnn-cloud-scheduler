from algorithms.min_min import MinMinScheduler
from dataset.models import Dataset
from env.state import SimulationState


class MaxMinScheduler(MinMinScheduler):
    """
    Implementation of the MinMin scheduling algorithm.

    MinMin is a simple scheduling algorithm that schedules the task with the smallest length
    on the VM that will complete the task the fastest.
    """

    def __init__(self) -> None:
        super().__init__("Max--Min")

    def select_task(self, dataset: Dataset, state: SimulationState) -> int:
        """Choose the task with the largest length."""
        largest_task = None
        largest_task_length = -float("inf")
        for task in dataset.tasks:
            if not state.task_states[task.id].is_ready:
                continue
            if task.length > largest_task_length:
                largest_task_length = task.length
                largest_task = task
        assert largest_task is not None

        return largest_task.id
