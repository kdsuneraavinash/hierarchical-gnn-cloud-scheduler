import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from scipy import stats

from dataset.models import Dataset, Host, Preference, Task, Vm, Workflow


@dataclass
class DatasetArgs:
    task_count: int
    """number of tasks"""
    max_host_count: int
    """number of hosts"""
    max_vm_count: int
    """number of VMs"""
    max_tasks_per_workflow: int
    """maximum number of tasks per workflow"""
    min_memory_gb: int = 1
    """minimum amount of RAM for a VM (in GB)"""
    max_memory_gb: int = 10
    """maximum amount of RAM for a VM (in GB)"""
    min_disk_gb: int = 1
    """minimum amount of disk for a VM (in GB)"""
    max_disk_gb: int = 10
    """maximum amount of disk for a VM (in GB)"""
    min_cpu_speed: int = 500
    """minimum CPU speed in MIPS"""
    max_cpu_speed: int = 5000
    """maximum CPU speed in MIPS"""
    task_length_dist: str = "normal"
    """task length distribution (normal, uniform, left_skewed, right_skewed)"""
    min_task_length: int = 500
    """minimum task length"""
    max_task_length: int = 100_000
    """maximum task length"""
    task_high_priority_probability: float = 0.5
    """probability that a task is high priority"""
    arrival_rate: float = 3
    """arrival rate of workflows/second (for dynamic arrival)"""
    makespan_preference: float | None = None
    """preference for optimizing makespan"""
    energy_consumption_preference: float | None = None
    """preference for optimizing energy consumption"""
    latency_score_preference: float | None = None
    """preference for optimizing energy consumption"""
    context: dict[str, str] = field(default_factory=dict)
    """additional context for the dataset"""


def generate_dataset(args: DatasetArgs, rng: np.random.RandomState) -> Dataset:
    """
    Generate a dataset with the specified arguments.
    """
    preference = generate_preference(args, rng)
    hosts = generate_hosts(args, rng)
    vms = generate_vms(args, rng)
    workflows = generate_workflows(args, rng)
    tasks = generate_tasks(args, rng)

    # Check if there are tasks that cannot be assigned to any VM
    # If so, they get their reqs reset (so they are assignable to any)
    for task in tasks:
        if any(vm.is_compatible(task) for vm in vms):
            continue
        task.req_memory_gb = args.min_memory_gb
        task.req_disk_gb = args.min_disk_gb

    dataset = Dataset(preference=preference, workflows=workflows, tasks=tasks, vms=vms, hosts=hosts)
    dataset.check_sanity()
    return dataset


# Generating Preference
# ----------------------------------------------------------------------------------------------------------------------


def generate_preference(args: DatasetArgs, rng: np.random.RandomState) -> Preference:
    makespan = args.makespan_preference
    energy_consumption = args.energy_consumption_preference
    latency_score = args.latency_score_preference
    # [0, 0, 0] = [.3, .3, .3]
    # [1, 0, 0] = [ 1,  0,  0]
    # [0, 1, 0] = [ 0,  1,  0]
    # [0, 0, 1] = [ 0,  0,  1]
    # [1, 1, 0] = [.5, .5,  0]
    # [1, 0, 1] = [.5,  0, .5]
    # [0, 1, 1] = [ 0, .5, .5]
    # [1, 1, 1] = [.3, .3, .3]
    if makespan is None:
        makespan = float(rng.randint(0, 2))
    if energy_consumption is None:
        energy_consumption = float(rng.randint(0, 2))
    if latency_score is None:
        latency_score = float(rng.randint(0, 2))

    if makespan + energy_consumption + latency_score == 0:
        makespan = energy_consumption = latency_score = 1
    total = makespan + energy_consumption + latency_score

    return Preference(
        makespan=makespan / total,
        energy_consumption=energy_consumption / total,
        latency_score=latency_score / total,
    )


# Generating Hosts
# ----------------------------------------------------------------------------------------------------------------------


def generate_hosts(args: DatasetArgs, rng: np.random.RandomState) -> list[Host]:
    """
    Generate a list of hosts with the specified number of hosts.
    Uses the host specifications from data/host_specs.json.
    """

    with open(Path(__file__).parent / "data" / "host_specs.json", "r") as f:
        available_hosts: list[dict[str, Any]] = json.load(f)

    host_count = rng.randint(2, args.max_host_count + 1)
    args.context["host_count"] = str(host_count)

    hosts: list[Host] = []
    for i in range(host_count):
        spec = available_hosts[rng.randint(0, len(available_hosts))]
        hosts.append(
            Host(
                id=i,
                cores=int(spec["cores"]),
                cpu_speed_mips=int(spec["cpu_speed_gips"] * 1e3),
                power_idle_watt=int(spec["power_idle_watt"]),
                power_peak_watt=int(spec["power_peak_watt"]),
            )
        )
    return hosts


# Generating and Allocating VMs
# ----------------------------------------------------------------------------------------------------------------------


def generate_vms(args: DatasetArgs, rng: np.random.RandomState) -> list[Vm]:
    """
    Generate a list of VMs with the specified number of VMs.
    """

    host_count = int(args.context["host_count"])
    vm_count = rng.randint(2, args.max_vm_count + 1)
    args.context["vm_count"] = str(vm_count)

    return [
        Vm(
            id=i,
            host_id=rng.randint(0, host_count),
            cpu_speed_mips=rng.randint(args.min_cpu_speed, args.max_cpu_speed + 1),
            memory_gb=rng.randint(args.min_memory_gb, args.max_memory_gb + 1),
            disk_gb=rng.randint(args.min_disk_gb, args.max_disk_gb + 1),
        )
        for i in range(vm_count)
    ]


# Generating Task Length
# ----------------------------------------------------------------------------------------------------------------------


def generate_task_length(args: DatasetArgs, rng: np.random.RandomState) -> Any | float:
    """
    Generate a random task length based on the specified method. <br/>
    Available methods: uniform, normal, left_skewed, right_skewed <br/>
    """

    low = args.min_task_length
    high = args.max_task_length

    if args.task_length_dist == "uniform":
        return stats.uniform.rvs(loc=low, scale=high - low, random_state=rng)

    # 99.7% of the data is within 3 standard deviations (by empirical rule)
    # so we set the standard deviation to be 1/6 of the range
    mean = (low + high) / 2
    std = (high - low) / 6

    value = low - 1
    while value < low or value > high:
        if args.task_length_dist == "normal":
            value = stats.norm.rvs(loc=mean, scale=std, random_state=rng)
        elif args.task_length_dist == "left_skewed":
            value = stats.skewnorm.rvs(-5, loc=mean, scale=std, random_state=rng)
        elif args.task_length_dist == "right_skewed":
            value = stats.skewnorm.rvs(5, loc=mean, scale=std, random_state=rng)
        else:
            raise ValueError(f"Invalid distribution: {args.task_length_dist}")
    return value


# Generating Task DAG
# ----------------------------------------------------------------------------------------------------------------------


def generate_dag(args: DatasetArgs, rng: np.random.RandomState) -> dict[int, set[int]]:
    """
    Generate a random Directed Acyclic Graph (DAG) using the G(n, p) model.
    The resulting graph is represented as an adjacency list. <br/>
    The resulting graph has n nodes, with node 0 being the starting node. <br/>
    If p is set to log(n + eps) / n where n is the number of generated nodes.
    """

    n = int(args.context["workflow_task_count"])
    if n == 1:
        return {0: set()}

    p = math.log(n + 0.1) / n

    nodes: dict[int, set[int]] = {i: set() for i in range(n)}
    start_nodes: set[int] = set(range(1, n))

    for i in range(1, n):
        for j in range(i + 1, n):
            if rng.random() < p:
                nodes[i].add(j)
                start_nodes.discard(j)

    for i in start_nodes:
        nodes[0].add(i)

    return nodes


# Generating Delay
# ----------------------------------------------------------------------------------------------------------------------


def generate_poisson_delay(args: DatasetArgs, rng: np.random.RandomState) -> Any | float:
    """
    Generate a random delay between workflows in a Poisson process.
    The delay is exponentially distributed with parameter lambda.
    """

    return stats.expon.rvs(scale=1 / args.arrival_rate, random_state=rng)


# Generating Tasks
# ----------------------------------------------------------------------------------------------------------------------


def generate_tasks(args: DatasetArgs, rng: np.random.RandomState) -> list[Task]:
    """
    Generate a list of tasks.
    """

    workflow_count = int(args.context["workflow_count"])
    workflow_task_counts = args.context["workflow_task_counts"].split(",")

    tasks: list[Task] = []
    for workflow_id in range(workflow_count):
        args.context["workflow_task_count"] = workflow_task_counts[workflow_id]
        dag = generate_dag(args, rng)
        tasks.extend(
            [
                Task(
                    id=len(tasks) + task_id,
                    workflow_id=workflow_id,
                    length=int(generate_task_length(args, rng)),
                    child_ids=[len(tasks) + child_id for child_id in child_ids],
                    req_memory_gb=rng.randint(args.min_memory_gb, args.max_memory_gb + 1),
                    req_disk_gb=rng.randint(args.min_disk_gb, args.max_disk_gb + 1),
                    priority=int(rng.random() <= args.task_high_priority_probability),
                )
                for task_id, child_ids in dag.items()
            ]
        )

    assert len(tasks) == args.task_count, f"Unexpected number of tasks generated: {len(tasks)}"
    return tasks


# Generating Workflows
# ----------------------------------------------------------------------------------------------------------------------


def generate_workflows(args: DatasetArgs, rng: np.random.RandomState) -> list[Workflow]:
    """
    Generate a list of workflows.
    """

    workflow_task_counts: list[int] = []
    while sum(workflow_task_counts) < args.task_count:
        task_count = rng.randint(1, args.max_tasks_per_workflow + 1)
        task_count_cap = args.task_count - sum(workflow_task_counts)
        workflow_task_counts.append(min(task_count, task_count_cap))

    args.context["workflow_count"] = str(len(workflow_task_counts))
    args.context["workflow_task_counts"] = str(",".join(map(str, workflow_task_counts)))

    arrival_time = 0
    workflows: list[Workflow] = []
    for workflow_id in range(len(workflow_task_counts)):
        arrival_time += int(generate_poisson_delay(args, rng))
        workflows.append(Workflow(id=workflow_id, arrival_time=arrival_time))

    return workflows


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    dataset = generate_dataset(tyro.cli(DatasetArgs), rng)
    json_data = json.dumps(dataset.to_json())
    print(json_data)
