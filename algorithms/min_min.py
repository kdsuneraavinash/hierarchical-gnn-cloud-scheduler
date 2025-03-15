from algorithms.base_greedy import BaseGreedyScheduler
from dataset.models import Dataset
from env.state import SimulationState


class MinMinScheduler(BaseGreedyScheduler):
    """
    Implementation of the MinMin scheduling algorithm.

    MinMin is a simple scheduling algorithm that schedules the task with the smallest length
    on the VM that will complete the task the fastest.
    """

    def __init__(self, name: str | None = None):
        super().__init__(name or "Min--Min")

    def select_task(self, dataset: Dataset, state: SimulationState) -> int:
        """Choose the task with the smallest length."""
        smallest_task = None
        smallest_task_length = float("inf")
        for task in dataset.tasks:
            if not state.task_states[task.id].is_ready:
                continue
            if task.length < smallest_task_length:
                smallest_task_length = task.length
                smallest_task = task
        assert smallest_task is not None

        return smallest_task.id

    def select_vm(self, task_id: int, dataset: Dataset, state: SimulationState) -> int:
        """Schedule the task on the VM that will complete the task the fastest."""
        # Select the best VM by comparing the completion times
        best_vm = None
        best_vm_completion_time = float("inf")
        for vm in dataset.vms:
            if not vm.is_compatible(dataset.tasks[task_id]):
                continue
            min_start_time: float = max(
                state.vm_states[vm.id].completion_time,
                max(
                    (
                        state.task_states[parent_task.id].completion_time
                        for parent_task in dataset.tasks
                        if task_id in parent_task.child_ids
                    ),
                    default=0,
                ),
            )

            completion_time = min_start_time + vm.execution_time(dataset.tasks[task_id])
            if best_vm_completion_time > completion_time:
                best_vm = vm
                best_vm_completion_time = completion_time

        if best_vm is None:
            raise Exception("No VM found for task")

        return best_vm.id
