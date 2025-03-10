from algorithms.base_dynamic import BaseDynamicScheduler
from dataset.models import Dataset
from env.state import SimulationState


class EnergyAwareSchduler(BaseDynamicScheduler):
    """
    Selects the task-VM pair that minimizes the weighted cost function.
    """

    def __init__(self, alpha: float = 1):
        super().__init__("Energy Aware")
        self.alpha = alpha

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
                energy = dataset.hosts[vm.host_id].active_power_consumption(task)
                completion_time = state.vm_states[vm.id].completion_time

                # Compute weighted cost function
                total_cost = self.alpha * energy + (1 - self.alpha) * completion_time
                if total_cost < min_weighted_cost:
                    min_weighted_cost = total_cost
                    best_task_id = task_id
                    best_vm_id = vm.id

        return best_task_id, best_vm_id
