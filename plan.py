import json
import random
import numpy as np
import torch
import tyro
from algorithms.drl_agent import DrlAgentScheduler
from algorithms.round_robin import RoundRobinScheduler
from dataset.generator import DatasetArgs, generate_dataset
from dataset.models import Solution


def main(seed: int, gnn: bool = False):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    dataset = generate_dataset(rng=np.random.RandomState(seed), args=DatasetArgs.create("real_world"), dataset_key="rw")

    # Schedule
    scheduler = RoundRobinScheduler()
    if gnn:
        scheduler = DrlAgentScheduler(
            name="Proposed",
            agent_type="gnn",
            model_path="logs/1762541449_gnn_real_world_[1][1][1]/model.pt",
        )
    assignments = scheduler.schedule(dataset)
    assignments = list(sorted(assignments, key=lambda a: a.start_time))
    running_vm = {assignment.task_id: assignment.vm_id for assignment in assignments}

    solution = Solution(dataset, assignments)
    print("Seed", seed)
    print("Makespan", solution.actual_makespan())
    print("Energy Consumption", solution.actual_energy_consumption())
    print("SLA Penalty", solution.actual_sla_penalty())
    print("Run Time", scheduler.run_time())
    print("Decision Latency", scheduler.decision_latency())

    # Dependencies from DAG
    dependencies = [set() for _ in range(len(dataset.tasks))]
    for task in dataset.tasks:
        for child_task_id in task.child_ids:
            dependencies[child_task_id].add(task.id)
    # Dependencies from running in same VM
    last_vm_task_id = [-1 for _ in range(len(dataset.vms))]
    for assignment in assignments:
        if last_vm_task_id[assignment.vm_id] != -1:
            dependencies[assignment.task_id].add(last_vm_task_id[assignment.vm_id])
        last_vm_task_id[assignment.vm_id] = assignment.task_id

    result = {
        "vms": [
            {
                "speed": dataset.vms[i].actual_cpu_speed_mips,
            }
            for i in range(len(dataset.vms))
        ],
        "tasks": [
            {
                "id": i,
                "len": int(dataset.tasks[i].actual_length),
                "deps": list(dependencies[i]),
                "vm": int(running_vm[i]),
            }
            for i in range(len(dataset.tasks))
        ],
    }

    print(json.dumps(result))


if __name__ == "__main__":
    tyro.cli(main)
