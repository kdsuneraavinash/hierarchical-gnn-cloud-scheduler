import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from dataset.generator import DatasetArgs
from dataset.models import Dataset, Task, Vm, Workflow
from dataset.real_world import generate_hosts, generate_poisson_delay, generate_preference


@dataclass
class SyntheticDatasetArgs(DatasetArgs):
    max_tasks_per_workflow: int = 0
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
    context: dict[str, str] = field(default_factory=dict)
    """additional context for the dataset"""


def generate_synthetic_dataset(key: str, args: SyntheticDatasetArgs, rng: np.random.RandomState) -> Dataset:
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

    dataset = Dataset(key, preference, workflows, tasks, vms, hosts, [])
    dataset.check_sanity()
    return dataset


# Generating and Allocating VMs
# ----------------------------------------------------------------------------------------------------------------------


def generate_vms(args: SyntheticDatasetArgs, rng: np.random.RandomState) -> list[Vm]:
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


def generate_task_length(args: SyntheticDatasetArgs, rng: np.random.RandomState) -> Any | float:
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


# Generating Tasks
# ----------------------------------------------------------------------------------------------------------------------


def generate_tasks(args: SyntheticDatasetArgs, rng: np.random.RandomState) -> list[Task]:
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


def generate_workflows(args: SyntheticDatasetArgs, rng: np.random.RandomState) -> list[Workflow]:
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
