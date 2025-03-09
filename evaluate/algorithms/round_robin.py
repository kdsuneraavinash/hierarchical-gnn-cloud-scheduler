from dataset.models import Dataset, Task, Vm
from evaluate.algorithms.base_ready import BaseReadyScheduler


class RoundRobinScheduler(BaseReadyScheduler):
    """
    Implementation of the Round Robin scheduling algorithm.

    Round Robin is a simple scheduling algorithm that schedules the tasks in a circular order.
    """

    vm_index: int = 0

    def __init__(self):
        super().__init__("Round--Robin")

    def select_task(self, ready_tasks: list[Task], dataset: Dataset) -> Task:
        """Choose the next task (with no preference)."""
        return ready_tasks[0]

    def select_vm(self, task: Task, dataset: Dataset) -> Vm:
        """Schedule the task on the next VM in the list."""
        while not dataset.vms[self.vm_index].is_compatible(task):
            self.vm_index = (self.vm_index + 1) % len(dataset.vms)

        selected_vm = dataset.vms[self.vm_index]
        # Move the cursor to the next VM
        self.vm_index = (self.vm_index + 1) % len(dataset.vms)

        return selected_vm
