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
    is_available: bool = True


@dataclass
class TaskState:
    is_ready: bool = False
    prev_task_id: int | None = None
    assigned_vm_id: int | None = None
    start_time: float = 0
    completion_time: float = 0


@dataclass
class SimulationState:
    task_states: list[TaskState]
    vm_states: list[VmState]
    time: float = 0
