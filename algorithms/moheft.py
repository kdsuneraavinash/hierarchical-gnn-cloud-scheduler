from dataclasses import dataclass
from typing import Callable
from algorithms.base_mo import BaseMoScheduler, SolutionStoreType
from dataset.models import Dataset
from env.utils import compute_task_makespan_ranks


class MoheftScheduler(BaseMoScheduler):
    """
    MOHEFT (Multi-Objective HEFT) implementation considering both makespan and energy consumption.
    Returns K tradeoff workflow schedules, each as a list of VmAssignment.
    """

    def __init__(self, solution_count: int, store: SolutionStoreType | None = None, index: int = 0):
        super().__init__("MOHEFT", store if store is not None else {}, index)
        self.solution_count = solution_count

    def get_pareto_solutions(self, dataset: Dataset) -> list[list[tuple[int, int]]]:
        task_rank = compute_task_makespan_ranks(dataset)
        sorted_tasks = sorted(dataset.tasks, key=lambda t: task_rank[t.id], reverse=True)

        candidate_schedules = [
            CandidateSchedule(
                assignments=[],
                vm_ready_times=[0] * len(dataset.vms),
                task_completion_times=[-1] * len(dataset.tasks),
                makespan=0,
                energy_consumption=0,
                latency_score=0,
            )
            for _ in range(self.solution_count)
        ]

        for task in sorted_tasks:
            new_candidate_schedules: list[CandidateSchedule] = []

            for schedule in candidate_schedules:
                for vm in dataset.vms:
                    if not vm.is_compatible(task):
                        continue

                    ready_time = schedule.vm_ready_times[vm.id]
                    parent_finish_times = [
                        schedule.task_completion_times[parent.id]
                        for parent in dataset.tasks
                        if task.id in parent.child_ids and schedule.task_completion_times[parent.id] >= 0
                    ]
                    parent_max = max(parent_finish_times, default=0.0)
                    start_time = max(ready_time, parent_max)
                    finish_time = start_time + (task.length / vm.cpu_speed_mips)
                    latency_score = start_time * task.priority
                    energy_cost = vm.host_id

                    new_schedule = CandidateSchedule(
                        assignments=schedule.assignments.copy(),
                        vm_ready_times=schedule.vm_ready_times.copy(),
                        task_completion_times=schedule.task_completion_times.copy(),
                        makespan=max(schedule.makespan, finish_time),
                        energy_consumption=schedule.energy_consumption + energy_cost,
                        latency_score=schedule.latency_score + latency_score,
                    )
                    new_schedule.assignments.append((task.id, vm.id))
                    new_schedule.vm_ready_times[vm.id] = finish_time
                    new_schedule.task_completion_times[task.id] = finish_time
                    new_candidate_schedules.append(new_schedule)

            self.sort_by_crowding_distance(new_candidate_schedules)
            candidate_schedules = new_candidate_schedules[: self.solution_count]

        solutions = [schedule.assignments for schedule in candidate_schedules]
        return solutions

    def sort_by_crowding_distance(self, schedules: list["CandidateSchedule"]):
        """
        Computes crowding distances for each schedule based on makespan, energy and latency.
        Extreme schedules (best and worst for each objective) are assigned infinite distance.
        """
        if not schedules:
            return schedules

        objective_fns: list[Callable[[CandidateSchedule], float]] = [
            lambda s: s.makespan,
            lambda s: s.energy_consumption,
            lambda s: s.latency_score,
        ]
        for obj_fn in objective_fns:
            schedules.sort(key=obj_fn)
            schedules[0].crowding_distance = float("inf")
            schedules[-1].crowding_distance = float("inf")

            obj_values = [obj_fn(s) for s in schedules]
            obj_min = min(obj_values)
            obj_max = max(obj_values)
            range_val = obj_max - obj_min if obj_max != obj_min else 1.0

            for i in range(1, len(schedules) - 1):
                distance = (obj_fn(schedules[i + 1]) - obj_fn(schedules[i - 1])) / range_val
                schedules[i].crowding_distance += distance

        schedules.sort(key=lambda s: s.crowding_distance, reverse=True)


@dataclass
class CandidateSchedule:
    assignments: list[tuple[int, int]]
    vm_ready_times: list[float]
    task_completion_times: list[float]
    makespan: float
    energy_consumption: float
    latency_score: float
    crowding_distance: float = 0
