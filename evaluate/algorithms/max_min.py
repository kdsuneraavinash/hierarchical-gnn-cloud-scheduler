from dataset.models import Dataset, Task
from evaluate.algorithms.min_min import MinMinScheduler


class MaxMinScheduler(MinMinScheduler):
    """
    Implementation of the MinMin scheduling algorithm.

    MinMin is a simple scheduling algorithm that schedules the task with the smallest length
    on the VM that will complete the task the fastest.
    """

    def __init__(self):
        super().__init__("Max--Min")

    def select_task(self, ready_tasks: list[Task], dataset: Dataset) -> Task:
        """Choose the task with the largest length."""
        largest_task = None
        largest_task_length = -float("inf")
        for task in ready_tasks:
            if task.length > largest_task_length:
                largest_task_length = task.length
                largest_task = task
        assert largest_task is not None

        return largest_task
