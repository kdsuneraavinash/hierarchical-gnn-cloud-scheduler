from algorithms.base_greedy import BaseGreedyScheduler
from dataset.models import Dataset
from env.state import SimulationState


class RoundRobinScheduler(BaseGreedyScheduler):
    """
    Implementation of the Round Robin scheduling algorithm.

    Round Robin is a simple scheduling algorithm that schedules the tasks in a circular order.
    """

    vm_index: int = 0

    def __init__(self) -> None:
        super().__init__("Round--Robin")

    def select_task(self, dataset: Dataset, state: SimulationState) -> int:
        """Choose the next task (with no preference)."""
        ready_tasks = [task for task in dataset.tasks if state.task_states[task.id].is_ready]
        return ready_tasks[0].id

    def select_vm(self, task_id: int, dataset: Dataset, state: SimulationState) -> int:
        """Schedule the task on the next VM in the list."""
        self.vm_index = (self.vm_index + 1) % len(dataset.vms)
        while not dataset.vms[self.vm_index].is_compatible(dataset.tasks[task_id]):
            self.vm_index = (self.vm_index + 1) % len(dataset.vms)
        return self.vm_index
