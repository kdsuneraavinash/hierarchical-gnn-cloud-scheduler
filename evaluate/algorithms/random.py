import random

from dataset.models import Dataset, Task, Vm
from evaluate.algorithms.base_ready import BaseReadyScheduler


class RandomScheduler(BaseReadyScheduler):
    def __init__(self):
        super().__init__("Random")

    def select_task(self, ready_tasks: list[Task], dataset: Dataset) -> Task:
        """Choose the next task (with no preference)."""
        return random.choice(ready_tasks)

    def select_vm(self, task: Task, dataset: Dataset) -> Vm:
        """Schedule the task on the next VM in the list."""
        compatible_vms = [vm for vm in dataset.vms if vm.is_compatible(task)]
        return random.choice(compatible_vms)
