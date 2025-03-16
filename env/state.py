from dataclasses import dataclass


@dataclass
class EnvState:
    task_states: list["TaskState"]
    vm_states: list["VmState"]
    task_dependencies: set[tuple[int, int]]


@dataclass
class VmState:
    assigned_task_id: int | None = None
    completion_time: float = 0


@dataclass
class TaskState:
    is_ready: bool = False
    assigned_vm_id: int | None = None
    start_time: float = 0
    completion_time: float = 0


@dataclass
class SimulationState:
    task_states: list[TaskState]
    vm_states: list[VmState]
    task_dependencies: set[tuple[int, int]]
