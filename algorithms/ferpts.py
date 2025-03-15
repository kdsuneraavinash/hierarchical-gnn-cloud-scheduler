from algorithms.base_static import BaseStaticScheduler
from dataset.models import Dataset


class FerptsScheduler(BaseStaticScheduler):
    """Implementation of the FERPTS (Fast and Energy-Aware Resource Provisioning and Task Scheduling) algorithm."""

    def __init__(self) -> None:
        super().__init__("FERPTS")

    def compute_assignments(self, dataset: Dataset) -> list[tuple[int, int]]:
        vm_ready_times: dict[int, float] = {vm.id: 0.0 for vm in dataset.vms}
        task_completion_times: dict[int, float] = {}
        assignments: list[tuple[int, int]] = []

        for task in dataset.tasks:
            best_vm, min_cost = None, float("inf")

            for vm in dataset.vms:
                if not vm.is_compatible(task):
                    continue
                ready_time = vm_ready_times[vm.id]
                parent_completion_times = [
                    task_completion_times[parent_task.id]
                    for parent_task in dataset.tasks
                    if task.id in parent_task.child_ids
                ]
                start_time = max(ready_time, max(parent_completion_times, default=0.0))
                finish_time = start_time + (task.length / vm.cpu_speed_mips)

                # Calculate energy cost based on runtime and power usage
                energy_cost = dataset.hosts[vm.host_id].active_power_consumption(task)

                # FERPTS minimizes both runtime and energy cost
                combined_cost = finish_time + energy_cost
                if combined_cost < min_cost:
                    best_vm = vm
                    min_cost = combined_cost

            if best_vm is None:
                raise Exception(f"Task {task.id} could not be scheduled on any VM.")

            vm_ready_times[best_vm.id] = min_cost
            task_completion_times[task.id] = min_cost
            assignments.append((task.id, best_vm.id))

        return assignments
