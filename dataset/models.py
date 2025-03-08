import dataclasses
from dataclasses import dataclass


@dataclass
class Workflow:
    id: int
    arrival_time: int


@dataclass
class Task:
    id: int
    workflow_id: int
    length: int
    req_memory_mb: int
    child_ids: list[int]


@dataclass
class Vm:
    id: int
    host_id: int
    cpu_speed_mips: int
    memory_mb: int
    disk_mb: int = -1
    bandwidth_mbps: int = -1
    vmm: str = "Xen"

    def is_compatible(self, task: Task):
        return self.memory_mb >= task.req_memory_mb

    def execution_time(self, task: Task):
        return task.length / self.cpu_speed_mips


@dataclass
class Host:
    id: int
    cores: int
    cpu_speed_mips: int
    power_idle_watt: int
    power_peak_watt: int
    memory_mb: int = -1
    disk_mb: int = -1
    bandwidth_mbps: int = -1

    def active_power_consumption(self, task: Task):
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

    def to_json(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict) -> "Dataset":
        workflows = [Workflow(**workflow) for workflow in data.pop("workflows")]
        tasks = [Task(**task) for task in data.pop("tasks")]
        vms = [Vm(**vm) for vm in data.pop("vms")]
        hosts = [Host(**host) for host in data.pop("hosts")]

        dataset = Dataset(workflows=workflows, tasks=tasks, vms=vms, hosts=hosts)
        dataset.check_sanity()
        return dataset

    def check_sanity(self):
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


@dataclass
class Solution:
    dataset: Dataset
    vm_assignments: list[VmAssignment]

    def to_json(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict) -> "Solution":
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
