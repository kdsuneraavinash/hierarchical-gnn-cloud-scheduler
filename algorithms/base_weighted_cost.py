from abc import ABC, abstractmethod

from algorithms.base_dynamic import BaseDynamicScheduler
from dataset.models import Dataset, Task, Vm
from env.state import SimulationState


class BaseWeightedCostScheduler(BaseDynamicScheduler, ABC):
    """
    Selects the task-VM pair that minimizes the weighted cost function.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def select_task_and_vm(self, dataset: Dataset, state: SimulationState) -> tuple[int, int]:
        # Get all ready tasks
        ready_tasks = [task.id for task in dataset.tasks if state.task_states[task.id].is_ready]

        best_task_id = -1
        best_vm_id = -1
        min_weighted_cost = float("inf")

        # Evaluate all (task, VM) pairs
        for task_id in ready_tasks:
            task = dataset.tasks[task_id]

            for vm in dataset.vms:
                if not vm.is_compatible(task):
                    continue
                total_cost = self.weighted_cost(task, vm, dataset, state)
                if total_cost < min_weighted_cost:
                    min_weighted_cost = total_cost
                    best_task_id = task_id
                    best_vm_id = vm.id

        return best_task_id, best_vm_id

    @abstractmethod
    def weighted_cost(self, task: Task, vm: Vm, dataset: Dataset, state: SimulationState) -> float:
        raise NotImplementedError()
