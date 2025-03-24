import dataclasses
import json
from dataclasses import dataclass
from hashlib import md5
from typing import Any

from constants import N_TASK, N_VM
from env.state import VmState


@dataclass
class Workflow:
    id: int
    arrival_time: int
    priority: float


@dataclass
class Task:
    id: int
    workflow_id: int
    length: int
    child_ids: list[int]
    req_memory_gb: float
    req_disk_gb: float
    actual_length: int


@dataclass
class Vm:
    id: int
    host_id: int
    cpu_speed_mips: int
    memory_gb: float
    disk_gb: float
    actual_cpu_speed_mips: int

    def is_compatible(self, task: Task, vm_state: VmState | None = None) -> bool:
        return (
            (vm_state is None or vm_state.is_available)
            and self.memory_gb >= task.req_memory_gb
            and self.disk_gb >= task.req_disk_gb
        )

    def execution_time(self, task: Task) -> float:
        return task.length / self.cpu_speed_mips

    def actual_execution_time(self, task: Task) -> float:
        return task.length / self.actual_cpu_speed_mips


@dataclass
class Host:
    id: int
    cores: int
    cpu_speed_mips: int
    power_idle_watt: int
    power_peak_watt: int
    actual_cpu_speed_mips: int
    actual_power_idle_watt: int
    actual_power_peak_watt: int

    @property
    def active_power_consumption_rate(self) -> float:
        return (self.power_peak_watt - self.power_idle_watt) / self.cpu_speed_mips

    @property
    def actual_active_power_consumption_rate(self) -> float:
        return (self.actual_power_peak_watt - self.actual_power_idle_watt) / self.actual_cpu_speed_mips

    def active_power_consumption(self, task: Task) -> float:
        return task.length * self.active_power_consumption_rate

    def actual_active_power_consumption(self, task: Task) -> float:
        return task.length * self.actual_active_power_consumption_rate


@dataclass
class VmEvent:
    class T:
        OFF = 0
        ON = 1

    time: float
    vm_id: int
    event_type: int  # 0 - Crash, 1 - Restore


@dataclass
class VmAssignment:
    task_id: int
    vm_id: int
    start_time: float


@dataclass
class Preference:
    makespan: float
    energy_consumption: float
    sla_penalty: float


@dataclass
class Dataset:
    key: str
    preference: Preference
    workflows: list[Workflow]
    tasks: list[Task]
    vms: list[Vm]
    hosts: list[Host]
    vm_events: list[VmEvent]

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Dataset":
        dataset = Dataset(
            key=data.pop("key"),
            preference=Preference(**data.pop("preference")),
            workflows=[Workflow(**workflow) for workflow in data.pop("workflows")],
            tasks=[Task(**task) for task in data.pop("tasks")],
            vms=[Vm(**vm) for vm in data.pop("vms")],
            hosts=[Host(**host) for host in data.pop("hosts")],
            vm_events=[VmEvent(**vm_event) for vm_event in data.pop("vm_events")],
        )
        dataset.check_sanity()
        return dataset

    def check_sanity(self, print_hash: bool = False) -> None:
        assert len(self.tasks) <= N_TASK, "There are more tasks than the buffer"
        assert len(self.vms) <= N_VM, "There are more vms than the buffer"

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
        # Check if all tasks are assignable to any VM
        for task in self.tasks:
            assert any(vm.is_compatible(task) for vm in self.vms), f"There are no VMs compatible with task {task=}"

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

    # --- actual metrics ---

    def actual_makespan(self) -> float:
        makespan: float = 0
        for assignment in self.vm_assignments:
            task = self.dataset.tasks[assignment.task_id]
            vm = self.dataset.vms[assignment.vm_id]
            makespan = max(makespan, assignment.start_time + vm.actual_execution_time(task))
        return makespan

    def actual_energy_consumption(self) -> float:
        energy_consumption: float = 0
        for assignment in self.vm_assignments:
            task = self.dataset.tasks[assignment.task_id]
            vm = self.dataset.vms[assignment.vm_id]
            host = self.dataset.hosts[vm.host_id]
            energy_consumption += host.actual_active_power_consumption(task)
        return energy_consumption

    def actual_sla_penalty(self) -> float:
        workflow_completion_times: list[float] = [0] * len(self.dataset.workflows)
        for assignment in self.vm_assignments:
            task = self.dataset.tasks[assignment.task_id]
            vm = self.dataset.vms[assignment.vm_id]
            finish_time = assignment.start_time + vm.actual_execution_time(task)
            workflow_completion_times[task.workflow_id] = max(finish_time, workflow_completion_times[task.workflow_id])

        return sum(
            workflow.priority * completion_time
            for workflow, completion_time in zip(self.dataset.workflows, workflow_completion_times)
        )
