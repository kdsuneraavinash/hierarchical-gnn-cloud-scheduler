import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from dataset.dag_gen import BaseDagGen, BranchParallelDagGen, EpigenomicsDagGen, InspiralDagGen
from dataset.generator import DatasetArgs
from dataset.models import Dataset, Host, Preference, Task, Vm, VmEvent, Workflow
from dataset.utils import random_list


epigenomics_dag_gen = EpigenomicsDagGen()
inspiral_dag_gen = InspiralDagGen()
branch_parallel_dag_gen = BranchParallelDagGen()


@dataclass
class RealWorldDatasetArgs(DatasetArgs):
    dag_structure: str = "Epigenomics"
    """dag structure to use"""
    vm_breakdowns: bool = False
    """existance of vm breakdowns where vms are not available"""
    vm_revival_max_gap: int = 20
    """gap between a vm breakdown and its revival"""
    vm_breakdown_max_gap: int = 5
    """gap between a vm revival and a new vm breakdown"""
    estimation_errors: bool = False
    """existance of estimation errors (actual when running is different)"""
    task_length_estimation_error: float = 0.1
    """error rate of the task length estimation"""
    cpu_speed_estimation_error: float = 0
    """error rate of the host/vm cpu speed estimation"""
    power_estimation_error: float = 0
    """error rate of the host/vm power estimation"""


def generate_real_world_dataset(key: str, args: RealWorldDatasetArgs, rng: np.random.RandomState) -> Dataset:
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
    disposable_vm_ids: set[int] = set(range(len(vms)))
    for task in tasks:
        compatible_vms = [vm for vm in vms if vm.is_compatible(task)]
        if len(compatible_vms) == 0:
            task.req_memory_gb = 0
            task.req_disk_gb = 0
        elif len(compatible_vms) == 1:
            if compatible_vms[0].id in disposable_vm_ids:
                disposable_vm_ids.remove(compatible_vms[0].id)

    # Create VM events of breakdown and revival based on non-critical VMs
    vm_events: list[VmEvent] = []
    if args.vm_breakdowns:
        vm_events.append(VmEvent(0, -1, -1))
        for _ in range(100):
            vm_id: int = rng.choice(list(disposable_vm_ids))
            vm_event_off_time = vm_events[-1].time + rng.randint(1, args.vm_breakdown_max_gap)
            vm_event_off = VmEvent(time=vm_event_off_time, vm_id=vm_id, event_type=VmEvent.T.OFF)
            vm_events.append(vm_event_off)
            vm_event_on_time = vm_events[-1].time + rng.randint(1, args.vm_revival_max_gap)
            vm_event_on = VmEvent(time=vm_event_on_time, vm_id=vm_id, event_type=VmEvent.T.ON)
            vm_events.append(vm_event_on)
        vm_events.pop(0)

    dataset = Dataset(key, preference, workflows, tasks, vms, hosts, vm_events)
    dataset.check_sanity()
    return dataset


# Generating Preference
# ----------------------------------------------------------------------------------------------------------------------


def generate_preference(args: DatasetArgs, rng: np.random.RandomState) -> Preference:
    makespan = args.makespan_preference
    energy_consumption = args.energy_consumption_preference
    sla_penalty = args.sla_penalty_preference
    if makespan + energy_consumption + sla_penalty == 0:
        makespan = energy_consumption = sla_penalty = 1
    max_pref = max(makespan, energy_consumption, sla_penalty)

    return Preference(
        makespan=makespan / max_pref,
        energy_consumption=energy_consumption / max_pref,
        sla_penalty=sla_penalty / max_pref,
    )


# Generating Hosts
# ----------------------------------------------------------------------------------------------------------------------


def generate_hosts(args: RealWorldDatasetArgs, rng: np.random.RandomState) -> list[Host]:
    with open(Path(__file__).parent / "data" / "host_specs.json", "r") as f:
        available_hosts: list[dict[str, Any]] = json.load(f)

    host_count = rng.randint(2, args.max_host_count + 1)
    args.context["host_count"] = str(host_count)

    hosts: list[Host] = []
    for i in range(host_count):
        spec = available_hosts[rng.randint(0, len(available_hosts))]
        actual_cpu_speed, est_cpu_speed = _estimate(float(spec["cpu_speed_gips"] * 1e3), "cpu", args, rng)
        actual_power_idle, est_power_idle = _estimate(float(spec["power_idle_watt"]), "power", args, rng)
        actual_power_peak, est_power_peak = _estimate(float(spec["power_peak_watt"]), "power", args, rng)

        hosts.append(
            Host(
                id=i,
                cores=int(spec["cores"]),
                cpu_speed_mips=int(est_cpu_speed),
                power_idle_watt=int(est_power_idle),
                power_peak_watt=int(est_power_peak),
                actual_cpu_speed_mips=int(actual_cpu_speed),
                actual_power_idle_watt=int(actual_power_idle),
                actual_power_peak_watt=int(actual_power_peak),
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

    vms: list[Vm] = []
    for i in range(vm_count):
        actual_cpu_speed, est_cpu_speed = _estimate(from_vm_dist("speed_mips"), "cpu", args, rng)
        vms.append(
            Vm(
                id=i,
                host_id=rng.randint(0, host_count),
                cpu_speed_mips=int(est_cpu_speed),
                memory_gb=from_vm_dist("memory_mb") / 1024,
                disk_gb=from_vm_dist("disk_gb"),
                actual_cpu_speed_mips=int(actual_cpu_speed),
            )
        )

    args.context["max_vm_memory_gb"] = str(max(vm.memory_gb for vm in vms))
    args.context["max_vm_disk_gb"] = str(max(vm.disk_gb for vm in vms))
    return vms


# Generating Task DAG
# ----------------------------------------------------------------------------------------------------------------------


def get_dag_gen(args: RealWorldDatasetArgs, rng: np.random.RandomState) -> BaseDagGen:
    if args.dag_structure == "Epigenomics":
        return epigenomics_dag_gen
    if args.dag_structure == "Inspiral":
        return inspiral_dag_gen
    if args.dag_structure == "BranchParallel":
        return branch_parallel_dag_gen
    raise ValueError("Unknown dag structure")


def min_dag_size(args: RealWorldDatasetArgs, rng: np.random.RandomState) -> int:
    return get_dag_gen(args, rng).min_size


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
    task_length_mean = float(task_specs["task_length_mean"])
    task_length_std = float(task_specs["task_length_std"])
    priority_production_prob = float(task_specs["priority_production_prob"])

    def task_length() -> float:
        value: float = 0
        while value <= 0:
            value = stats.norm.rvs(loc=task_length_mean, scale=task_length_std, random_state=rng)
        return value

    tasks: list[Task] = []
    for workflow_id in range(workflow_count):
        dag = get_dag_gen(args, rng).generate(tasks_per_workflow[workflow_id], rng)
        task_offset = len(tasks)
        for task_id, child_ids in dag.items():
            actual_task_length, est_task_length = _estimate(task_length(), "task_length", args, rng)
            tasks.append(
                Task(
                    id=task_offset + task_id,
                    workflow_id=workflow_id,
                    length=int(est_task_length),
                    child_ids=[task_offset + child_id for child_id in child_ids],
                    req_memory_gb=rng.uniform(low=0, high=max_vm_memory_gb),
                    req_disk_gb=rng.uniform(low=0, high=max_vm_disk_gb),
                    actual_length=int(actual_task_length),
                    priority=int(rng.random() < priority_production_prob),
                )
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

    min_size = min_dag_size(args, rng)
    max_workflow_count = args.task_count // min_size
    workflow_count = rng.randint(1, max_workflow_count + 1)
    tasks_per_workflow = random_list(workflow_count, args.task_count, rng, min_value=min_size)

    args.context["workflow_count"] = str(workflow_count)
    args.context["tasks_per_workflow"] = ",".join(map(str, tasks_per_workflow))

    arrival_time = 0
    workflows: list[Workflow] = []
    for workflow_id in range(workflow_count):
        arrival_time += int(generate_poisson_delay(args, rng))
        workflows.append(Workflow(id=workflow_id, arrival_time=arrival_time))

    return workflows


# Utils
# ----------------------------------------------------------------------------------------------------------------------


def _estimate(actual: float, metric: str, args: RealWorldDatasetArgs, rng: np.random.RandomState):
    estimation = actual
    if args.estimation_errors:
        error_rate = 0
        if metric == "cpu":
            error_rate = args.cpu_speed_estimation_error
        elif metric == "power":
            error_rate = args.power_estimation_error
        elif metric == "task_length":
            error_rate = args.task_length_estimation_error
        else:
            raise ValueError(f"Unknown metric: {metric}")

        error = (2 * rng.random() - 1) * error_rate
        estimation += int(error * estimation)
    return actual, estimation
