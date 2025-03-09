from dataset.models import Dataset
from env.simulation import SimulationState
from evaluate.algorithms.base_greedy import BaseGreedyScheduler


class RoundRobinScheduler(BaseGreedyScheduler):
    """
    Implementation of the Round Robin scheduling algorithm.

    Round Robin is a simple scheduling algorithm that schedules the tasks in a circular order.
    """

    vm_index: int = 0

    def __init__(self):
        super().__init__("Round--Robin")

    def select_task(self, dataset: Dataset, state: SimulationState) -> int:
        """Choose the next task (with no preference)."""
        ready_tasks = [task for task in dataset.tasks if state.task_states[task.id].is_ready]
        return ready_tasks[0].id

    def select_vm(self, task_id: int, dataset: Dataset, state: SimulationState) -> int:
        """Schedule the task on the next VM in the list."""
        while not dataset.vms[self.vm_index].is_compatible(dataset.tasks[task_id]):
            self.vm_index = (self.vm_index + 1) % len(dataset.vms)

        selected_vm = dataset.vms[self.vm_index]
        # Move the cursor to the next VM
        self.vm_index = (self.vm_index + 1) % len(dataset.vms)

        return selected_vm.id
