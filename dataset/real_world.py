import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
import tyro

from dataset.dag_gen import MIN_DAG_SIZE, EpigenomicsDagGen, InspiralDagGen
from dataset.generator import DatasetArgs
from dataset.models import Dataset, Host, Preference, Task, Vm, Workflow
from dataset.utils import random_list


epigenomics_dag_gen = EpigenomicsDagGen()
inspiral_dag_gen = InspiralDagGen()


@dataclass
class RealWorldDatasetArgs(DatasetArgs):
    dag_structure: str = "Epigenomics"
    """dag structure to use"""


def generate_real_world_dataset(args: RealWorldDatasetArgs, rng: np.random.RandomState) -> Dataset:
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
        task.req_memory_gb = 0
        task.req_disk_gb = 0

    dataset = Dataset(preference=preference, workflows=workflows, tasks=tasks, vms=vms, hosts=hosts)
    dataset.check_sanity()
    return dataset


# Generating Preference
# ----------------------------------------------------------------------------------------------------------------------


def generate_preference(args: DatasetArgs, rng: np.random.RandomState) -> Preference:
    makespan = args.makespan_preference
    energy_consumption = args.energy_consumption_preference
    latency_score = args.latency_score_preference
    if makespan + energy_consumption + latency_score == 0:
        makespan = energy_consumption = latency_score = 1
    max_pref = max(makespan, energy_consumption, latency_score)

    return Preference(
        makespan=makespan / max_pref,
        energy_consumption=energy_consumption / max_pref,
        latency_score=latency_score / max_pref,
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


def generate_vms(args: RealWorldDatasetArgs, rng: np.random.RandomState) -> list[Vm]:
    """
    Generate a list of VMs with the specified number of VMs.
    """

    host_count = int(args.context["host_count"])
    vm_count = rng.randint(2, args.max_vm_count + 1)
    args.context["vm_count"] = str(vm_count)

    with open(Path(__file__).parent / "data" / "vm_specs.json", "r") as f:
        vm_specs: dict[str, Any] = json.load(f)

    def from_vm_dist(key: str) -> int:
        values = list(map(int, vm_specs[key].keys()))
        probabilities = list(map(float, vm_specs[key].values()))
        return int(rng.choice(values, p=probabilities))

    vms = [
        Vm(
            id=i,
            host_id=rng.randint(0, host_count),
            cpu_speed_mips=from_vm_dist("speed_mips"),
            memory_gb=from_vm_dist("memory_gb") / 1024,
            disk_gb=from_vm_dist("disk_gb"),
        )
        for i in range(vm_count)
    ]

    args.context["max_vm_memory_gb"] = str(max(vm.memory_gb for vm in vms))
    args.context["max_vm_disk_gb"] = str(max(vm.disk_gb for vm in vms))
    return vms


# Generating Task DAG
# ----------------------------------------------------------------------------------------------------------------------


def generate_dag_pegasus(args: RealWorldDatasetArgs, rng: np.random.RandomState) -> dict[int, set[int]]:
    workflow_task_count = int(args.context["workflow_task_count"])
    if args.dag_structure == "Epigenomics":
        return epigenomics_dag_gen.generate(workflow_task_count, rng)
    if args.dag_structure == "Inspiral":
        return inspiral_dag_gen.generate(workflow_task_count, rng)
    raise ValueError("Unknown dag structure")


# Generating Tasks
# ----------------------------------------------------------------------------------------------------------------------


def generate_tasks(args: RealWorldDatasetArgs, rng: np.random.RandomState) -> list[Task]:
    """
    Generate a list of tasks.
    """

    workflow_count = int(args.context["workflow_count"])
    max_vm_memory_gb = float(args.context["max_vm_memory_gb"])
    max_vm_disk_gb = float(args.context["max_vm_disk_gb"])
    tasks_per_workflow = list(map(int, args.context["tasks_per_workflow"].split(",")))

    with open(Path(__file__).parent / "data" / "task_specs.json", "r") as f:
        task_specs: dict[str, Any] = json.load(f)

    priority_production_prob = float(task_specs["priority_production_prob"])
    task_length_mean = float(task_specs["task_length_mean"])
    task_length_std = float(task_specs["task_length_std"])

    def task_length() -> float:
        value: float = 0
        while value <= 0:
            value = stats.norm.rvs(loc=task_length_mean, scale=task_length_std, random_state=rng)
        return value

    tasks: list[Task] = []
    for workflow_id in range(workflow_count):
        args.context["workflow_task_count"] = str(tasks_per_workflow[workflow_id])
        dag = generate_dag_pegasus(args, rng)
        tasks.extend(
            [
                Task(
                    id=len(tasks) + task_id,
                    workflow_id=workflow_id,
                    length=int(task_length()),
                    child_ids=[len(tasks) + child_id for child_id in child_ids],
                    req_memory_gb=rng.uniform(low=0, high=max_vm_memory_gb),
                    req_disk_gb=rng.uniform(low=0, high=max_vm_disk_gb),
                    priority=int(rng.random() <= priority_production_prob),
                )
                for task_id, child_ids in dag.items()
            ]
        )

    return tasks


# Generating Delay
# ----------------------------------------------------------------------------------------------------------------------


def generate_poisson_delay(args: DatasetArgs, rng: np.random.RandomState) -> Any | float:
    """
    Generate a random delay between workflows in a Poisson process.
    The delay is exponentially distributed with parameter lambda.
    """

    return stats.expon.rvs(scale=1 / args.arrival_rate, random_state=rng)


# Generating Workflows
# ----------------------------------------------------------------------------------------------------------------------


def generate_workflows(args: RealWorldDatasetArgs, rng: np.random.RandomState) -> list[Workflow]:
    """
    Generate a list of workflows.
    """

    max_workflow_count = args.task_count // MIN_DAG_SIZE
    workflow_count = rng.randint(1, max_workflow_count + 1)
    tasks_per_workflow = random_list(workflow_count, args.task_count, rng, min_value=MIN_DAG_SIZE)

    args.context["workflow_count"] = str(workflow_count)
    args.context["tasks_per_workflow"] = ",".join(map(str, tasks_per_workflow))

    arrival_time = 0
    workflows: list[Workflow] = []
    for workflow_id in range(workflow_count):
        arrival_time += int(generate_poisson_delay(args, rng))
        workflows.append(Workflow(id=workflow_id, arrival_time=arrival_time))

    return workflows


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    dataset = generate_real_world_dataset(tyro.cli(RealWorldDatasetArgs), rng)
    json_data = json.dumps(dataset.to_json())
    print(json_data)
