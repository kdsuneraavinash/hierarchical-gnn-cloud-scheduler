from collections import deque
from typing import Any
import numpy as np
from pymoo.core.problem import ElementwiseProblem

from dataset.models import Dataset
from env.simulation import Simulation


def decode_pymoo_x(dataset: Dataset, x: np.ndarray[tuple[int, ...], Any]) -> list[tuple[int, int]]:
    num_tasks = len(dataset.tasks)
    ranked_order = list(map(int, np.argsort(x[:num_tasks])))
    vm_assignments = list(map(int, x[num_tasks:]))

    # Use Kahn’s Algorithm (BFS) to Sort Tasks with Dependencies
    in_degree = {task_id: 0 for task_id in range(num_tasks)}
    for parent_id in range(num_tasks):
        for child in dataset.tasks[parent_id].child_ids:
            in_degree[child] += 1
    queue = deque([task for task in ranked_order if in_degree[task] == 0])
    valid_order: list[int] = []
    while queue:
        task_id = queue.popleft()
        valid_order.append(task_id)
        for child_id in dataset.tasks[task_id].child_ids:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    assert len(valid_order) == num_tasks, "Are there any cycles?"

    # Set the VM to the nearest compatible VM
    valid_vm_assignments: list[int] = []
    for task_id, vm_id in zip(valid_order, vm_assignments):
        while not dataset.vms[vm_id].is_compatible(dataset.tasks[task_id]):
            vm_id = (vm_id + 1) % len(dataset.vms)
        valid_vm_assignments.append(vm_id)

    return list(zip(valid_order, valid_vm_assignments))


class PyMooOptimizationProblem(ElementwiseProblem):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.num_tasks = len(dataset.tasks)
        self.num_vms = len(dataset.vms)

        # The solution will have 2 parts
        # [0, 1, ..., Nt - 1,           Nt, Nt + 1, ..., 2*Nt - 1]
        # order of task selection       assigned vm
        super().__init__(
            n_var=self.num_tasks + self.num_tasks,
            n_obj=3,
            xl=np.hstack([np.zeros(self.num_tasks), np.zeros(self.num_tasks)]),
            xu=np.hstack([np.full(self.num_tasks, self.num_tasks - 1), np.full(self.num_tasks, self.num_vms - 1)]),
        )

    def _evaluate(self, x: np.ndarray[tuple[int, ...], Any], out, *args, **kwargs):
        assignments = decode_pymoo_x(self.dataset, x)

        simulation = Simulation(self.dataset)
        for task_id, vm_id in assignments:
            error, done = simulation.assign_vm(task_id, vm_id)
            if error:
                raise ValueError(error)

        out["F"] = [
            simulation.makespan(),
            simulation.total_energy_consumption(),
            simulation.total_sla_penalty(),
        ]
