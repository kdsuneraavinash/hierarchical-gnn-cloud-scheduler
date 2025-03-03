import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from scipy import stats

from dataset.models import Dataset, Host, Task, Vm, Workflow


@dataclass
class DatasetArgs:
    seed: int | None = 42
    """random seed"""
    host_count: int = 2
    """number of hosts"""
    vm_count: int = 3
    """number of VMs"""
    max_memory_gb: int = 10
    """maximum amount of RAM for a VM (in GB)"""
    min_cpu_speed: int = 500
    """minimum CPU speed in MIPS"""
    max_cpu_speed: int = 5000
    """maximum CPU speed in MIPS"""
    workflow_count: int = 5
    """number of workflows"""
    dag_method: str = "gnp"
    """DAG generation method (pegasus, gnp)"""
    gnp_min_n: int = 1
    """minimum number of tasks per workflow (for G(n,p) method)"""
    gnp_max_n: int = 10
    """maximum number of tasks per workflow (for G(n,p) method)"""
    task_length_dist: str = "normal"
    """task length distribution (normal, uniform, left_skewed, right_skewed)"""
    min_task_length: int = 500
    """minimum task length"""
    max_task_length: int = 100_000
    """maximum task length"""
    arrival_rate: float = 3
    """arrival rate of workflows/second (for dynamic arrival)"""
    context: dict[str, str] = field(default_factory=dict)
    """additional context for the dataset"""


def generate_dataset(args: DatasetArgs) -> Dataset:
    """
    Generate a dataset with the specified arguments.
    """
    rng = np.random.RandomState(args.seed)

    hosts = generate_hosts(args, rng)
    vms = generate_vms(args, rng)
    tasks = generate_tasks(args, rng)
    workflows = generate_workflows(args, rng)

    dataset = Dataset(workflows=workflows, tasks=tasks, vms=vms, hosts=hosts)
    dataset.check_sanity()
    return dataset


# Generating Hosts
# ----------------------------------------------------------------------------------------------------------------------


def generate_hosts(args: DatasetArgs, rng: np.random.RandomState) -> list[Host]:
    """
    Generate a list of hosts with the specified number of hosts.
    Uses the host specifications from data/host_specs.json.
    """

    with open(Path(__file__).parent / "data" / "host_specs.json", "r") as f:
        available_hosts: list = json.load(f)

    hosts: list[Host] = []
    for i in range(args.host_count):
        spec = available_hosts[rng.randint(0, len(available_hosts))]
        hosts.append(
            Host(
                id=i,
                cores=int(spec["cores"]),
                cpu_speed_mips=int(spec["cpu_speed_gips"] * 1e3),
                memory_mb=int(spec["memory_gb"] * 1024),
                disk_mb=int(spec["disk_tb"] * 1e6),
                bandwidth_mbps=int(spec["bandwidth_gbps"] * 1024),
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

    vms: list[Vm] = []
    for i in range(args.vm_count):
        ram_mb = rng.randint(1, args.max_memory_gb + 1) * 1024
        cpu_speed = rng.randint(args.min_cpu_speed, args.max_cpu_speed + 1)
        host_id = rng.randint(0, args.host_count)
        vms.append(Vm(i, host_id, cpu_speed, memory_mb=ram_mb, disk_mb=1024, bandwidth_mbps=50, vmm="Xen"))

    args.context["vm_max_memory"] = str(max(vm.memory_mb for vm in vms))
    return vms


# Generating Memory
# ----------------------------------------------------------------------------------------------------------------------


def generate_task_memory(args: DatasetArgs, rng: np.random.RandomState) -> int:
    vm_max_memory = int(args.context["vm_max_memory"])
    return (1 + rng.randint(0, vm_max_memory // 1024)) * 1024


# Generating Task Length
# ----------------------------------------------------------------------------------------------------------------------


def generate_task_length(args: DatasetArgs, rng: np.random.RandomState) -> float:
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
    if args.task_length_dist == "normal":
        method = lambda: stats.norm.rvs(loc=mean, scale=std, random_state=rng)
    elif args.task_length_dist == "left_skewed":
        method = lambda: stats.skewnorm.rvs(-5, loc=mean, scale=std, random_state=rng)
    elif args.task_length_dist == "right_skewed":
        method = lambda: stats.skewnorm.rvs(5, loc=mean, scale=std, random_state=rng)
    else:
        raise ValueError(f"Invalid distribution: {args.task_length_dist}")

    value = low - 1
    while value < low or value > high:
        value = method()
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

    n = rng.randint(args.gnp_min_n, args.gnp_max_n + 1)
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


def generate_poisson_delay(args: DatasetArgs, rng: np.random.RandomState) -> float:
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

    tasks: list[Task] = []
    for workflow_id in range(args.workflow_count):
        dag = generate_dag(args, rng)
        tasks.extend(
            [
                Task(
                    id=len(tasks) + task_id,
                    workflow_id=workflow_id,
                    length=int(generate_task_length(args, rng)),
                    req_memory_mb=generate_task_memory(args, rng),
                    child_ids=[len(tasks) + child_id for child_id in child_ids],
                )
                for task_id, child_ids in dag.items()
            ]
        )

    return tasks


# Generating Workflows
# ----------------------------------------------------------------------------------------------------------------------


def generate_workflows(args: DatasetArgs, rng: np.random.RandomState) -> list[Workflow]:
    """
    Generate a list of workflows.
    """

    arrival_time = 0
    workflows: list[Workflow] = []
    for workflow_id in range(args.workflow_count):
        arrival_time += int(generate_poisson_delay(args, rng))
        workflows.append(Workflow(id=workflow_id, arrival_time=arrival_time))

    return workflows


if __name__ == "__main__":
    dataset = generate_dataset(tyro.cli(DatasetArgs))
    json_data = json.dumps(dataset.to_json())
    print(json_data)
