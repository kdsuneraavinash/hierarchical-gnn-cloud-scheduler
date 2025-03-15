from algorithms.base_static import BaseStaticScheduler
from dataset.models import Dataset, Task, Vm


class HeftScheduler(BaseStaticScheduler):
    """Implementation of the HEFT (Heterogeneous Earliest Finish Time) algorithm."""

    def __init__(self) -> None:
        super().__init__("HEFT")

    def compute_assignments(self, dataset: Dataset) -> list[tuple[int, int]]:
        # Compute task priorities based on upward rank
        task_rank = self.compute_task_priorities(dataset.tasks, dataset.vms)
        sorted_tasks = sorted(dataset.tasks, key=lambda t: task_rank[t.id], reverse=True)

        vm_ready_times: dict[int, float] = {vm.id: 0.0 for vm in dataset.vms}
        task_completion_times: dict[int, float] = {}
        assignments: list[tuple[int, int]] = []
        for task in sorted_tasks:
            best_vm, earliest_finish_time = None, float("inf")

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
                finish_time = start_time + vm.execution_time(task)
                if finish_time < earliest_finish_time:
                    best_vm = vm
                    earliest_finish_time = finish_time

            if best_vm is None:
                raise Exception(f"Task {task.id} could not be scheduled on any VM.")

            vm_ready_times[best_vm.id] = earliest_finish_time
            task_completion_times[task.id] = earliest_finish_time

            assignments.append((task.id, best_vm.id))

        return assignments

    def compute_task_priorities(self, tasks: list[Task], vms: list[Vm]) -> dict[int, float]:
        """Compute task priorities based on upward rank."""

        average_vm_speed = sum(vm.cpu_speed_mips for vm in vms) / len(vms)
        task_rank: dict[int, float] = {}

        def compute_upward_rank(task: Task) -> float:
            if task.id in task_rank:
                return task_rank[task.id]

            child_ranks = [compute_upward_rank(child_task) for child_task in tasks if child_task.id in task.child_ids]
            child_rank_max = max(child_ranks, default=0)
            task_rank[task.id] = (task.length / average_vm_speed) + child_rank_max
            return task_rank[task.id]

        for _task in tasks:
            compute_upward_rank(_task)

        return task_rank
