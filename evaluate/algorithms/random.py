import random

from dataset.models import Dataset, VmAssignment
from evaluate.algorithms.base import BaseScheduler, to_assignments


class RandomScheduler(BaseScheduler):
    def __init__(self):
        super().__init__("Random")

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        assignments: list[tuple[int, int]] = []
        for task in dataset.tasks:
            compatible_vm_ids: list[int] = []
            for vm in dataset.vms:
                if vm.is_compatible(task):
                    compatible_vm_ids.append(vm.id)
            assignments.append((task.id, random.choice(compatible_vm_ids)))
        return to_assignments(dataset, assignments)
