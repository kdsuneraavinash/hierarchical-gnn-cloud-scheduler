from dataset.models import Dataset, Task, Vm
from evaluate.algorithms.base_ready import BaseReadyScheduler


class MinMinScheduler(BaseReadyScheduler):
    """
    Implementation of the MinMin scheduling algorithm.

    MinMin is a simple scheduling algorithm that schedules the task with the smallest length
    on the VM that will complete the task the fastest.
    """

    def __init__(self, name: str | None = None):
        super().__init__(name or "Min--Min")

    def select_task(self, ready_tasks: list[Task], dataset: Dataset) -> Task:
        """Choose the task with the smallest length."""
        smallest_task = None
        smallest_task_length = float("inf")
        for task in ready_tasks:
            if task.length < smallest_task_length:
                smallest_task_length = task.length
                smallest_task = task
        assert smallest_task is not None

        return smallest_task

    def select_vm(self, task: Task, dataset: Dataset) -> Vm:
        """Schedule the task on the VM that will complete the task the fastest."""
        # Select the best VM by comparing the completion times
        best_vm = None
        best_vm_completion_time = float("inf")
        for vm in dataset.vms:
            if not vm.is_compatible(task):
                continue

            min_start_time: float = max(
                (
                    self.task_states[parent_task.id].completion_time
                    for parent_task in dataset.tasks
                    if task.id in parent_task.child_ids
                ),
                default=0,
            )

            completion_time = max(self.vm_states[vm.id].completion_time, min_start_time) + vm.execution_time(task)
            if best_vm_completion_time > completion_time:
                best_vm = vm
                best_vm_completion_time = completion_time

        if best_vm is None:
            raise Exception("No VM found for task")

        return best_vm
