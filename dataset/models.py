import dataclasses
import json
from dataclasses import dataclass
from hashlib import md5
from typing import Any


@dataclass
class Workflow:
    id: int
    arrival_time: int


@dataclass
class Task:
    id: int
    workflow_id: int
    length: int
    child_ids: list[int]
    req_cpu_speed_mips: int
    req_memory_gb: int
    req_disk_gb: int
    req_bandwidth_mbps: int
    req_gpu: bool
    priority: float


@dataclass
class Vm:
    id: int
    host_id: int
    cpu_speed_mips: int
    memory_gb: int
    disk_gb: int
    bandwidth_mbps: int
    has_gpu: bool

    def is_fully_compatible(self, task: Task) -> bool:
        return self.compatibility(task) > 0

    def compatibility(self, task: Task) -> float:
        score: float = 1
        if self.memory_gb < task.req_memory_gb:
            score -= 0.4
        if self.cpu_speed_mips < task.req_cpu_speed_mips:
            score -= 0.3
        if self.disk_gb < task.req_disk_gb:
            score -= 0.2
        if self.bandwidth_mbps < task.req_bandwidth_mbps:
            score -= 0.1
        if (not self.has_gpu) and task.req_gpu:
            score -= 0.3
        return max(score, 0.0)

    def penalty(self, task: Task) -> float:
        return (1 - self.compatibility(task)) * task.priority

    def execution_time(self, task: Task) -> float:
        return task.length / self.cpu_speed_mips


@dataclass
class Host:
    id: int
    cores: int
    cpu_speed_mips: int
    power_idle_watt: int
    power_peak_watt: int
    memory_gb: int = -1
    disk_gb: int = -1
    bandwidth_mbps: int = -1

    def active_power_consumption(self, task: Task) -> float:
        return task.length * (self.power_peak_watt - self.power_idle_watt) / self.cpu_speed_mips


@dataclass
class VmAssignment:
    task_id: int
    vm_id: int
    start_time: float


@dataclass
class Dataset:
    workflows: list[Workflow]
    tasks: list[Task]
    vms: list[Vm]
    hosts: list[Host]

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Dataset":
        workflows = [Workflow(**workflow) for workflow in data.pop("workflows")]
        tasks = [Task(**task) for task in data.pop("tasks")]
        vms = [Vm(**vm) for vm in data.pop("vms")]
        hosts = [Host(**host) for host in data.pop("hosts")]

        dataset = Dataset(workflows=workflows, tasks=tasks, vms=vms, hosts=hosts)
        dataset.check_sanity()
        return dataset

    def check_sanity(self, print_hash: bool = False) -> None:
        # Sanity check - we should be able to use index and id interchangeably
        for i, workflow in enumerate(self.workflows):
            assert workflow.id == i, f"Sanity Check Failed: workflow ID mismatch, {workflow=} in index {i}"
        for i, task in enumerate(self.tasks):
            assert task.id == i, f"Sanity Check Failed: task ID mismatch, {task=} in index {i}"
        for i, vm in enumerate(self.vms):
            assert vm.id == i, f"Sanity Check Failed: vm ID mismatch, {vm=} in index {i}"
        for i, host in enumerate(self.hosts):
            assert host.id == i, f"Sanity Check Failed: host ID mismatch, {host=} in index {i}"
        # Check if all child ids are greater than the task id
        for task in self.tasks:
            for child_id in task.child_ids:
                assert child_id > task.id, f"Sanity Check Failed: {task=} has child id {child_id} less than task id"

        # It is possible to output the dataset hash for debug purposes
        if print_hash:
            obj_str = json.dumps(self.to_json())
            obj_md5 = md5(obj_str.encode("utf-8"))
            print("Dataset hash:", obj_md5.hexdigest())


@dataclass
class Solution:
    dataset: Dataset
    vm_assignments: list[VmAssignment]

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Solution":
        dataset = Dataset.from_json(data.pop("dataset"))
        vm_assignments = [VmAssignment(**vm_assignment) for vm_assignment in data.pop("vm_assignments")]
        return Solution(dataset=dataset, vm_assignments=vm_assignments)

    def makespan(self) -> float:
        makespan: float = 0
        for assignment in self.vm_assignments:
            task = self.dataset.tasks[assignment.task_id]
            vm = self.dataset.vms[assignment.vm_id]
            makespan = max(makespan, assignment.start_time + vm.execution_time(task))
        return makespan

    def energy_consumption(self) -> float:
        energy_consumption: float = 0
        for assignment in self.vm_assignments:
            task = self.dataset.tasks[assignment.task_id]
            vm = self.dataset.vms[assignment.vm_id]
            host = self.dataset.hosts[vm.host_id]
            energy_consumption += host.active_power_consumption(task)
        return energy_consumption

    def sla_penalty(self) -> float:
        sla_penalty: float = 0
        for assignment in self.vm_assignments:
            task = self.dataset.tasks[assignment.task_id]
            vm = self.dataset.vms[assignment.vm_id]
            sla_penalty += vm.penalty(task)
        return sla_penalty
