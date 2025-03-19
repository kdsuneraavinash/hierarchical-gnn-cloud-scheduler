import numpy as np
from dataset.models import Dataset, VmAssignment
from env.state import SimulationState, TaskState, VmState
from env.utils import task_completion_time_est, task_energy_consumption_est, task_latency_score_est


class Simulation:
    dataset: Dataset
    state: SimulationState

    # Initialization
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, dataset: Dataset):
        # Initial states of VMs
        vm_states = [VmState() for _ in dataset.vms]

        # Initialize task states and dependencies
        task_states = [TaskState(is_ready=True) for _ in dataset.tasks]
        task_dependencies = {(task.id, child_id) for task in dataset.tasks for child_id in task.child_ids}

        # Mark child tasks as not ready
        for _, child_id in task_dependencies:
            task_states[child_id].is_ready = False

        # Map to the state
        self.dataset = dataset
        self.state = SimulationState(task_states, vm_states, task_dependencies)

    # Assignment
    # ------------------------------------------------------------------------------------------------------------------

    def assign_vm(self, task_id: int, vm_id: int) -> tuple[str | None, bool]:
        # Checks for action
        if not (0 <= task_id < len(self.state.task_states)):
            return f"{task_id=} {vm_id=}: Invalid task (out of range)", True
        if not (0 <= vm_id < len(self.state.vm_states)):
            return f"{task_id=} {vm_id=}: Invalid vm (out of range)", True
        if self.state.task_states[task_id].assigned_vm_id is not None:
            return f"{task_id=} {vm_id=}: Already scheduled task", True
        if not self.state.task_states[task_id].is_ready:
            return f"{task_id=} {vm_id=}: Not ready task", True
        if not self.dataset.vms[vm_id].is_compatible(self.dataset.tasks[task_id]):
            return f"{task_id=} {vm_id=}: Not compatible", True

        # Convert to numpy arrays
        processing_time = self.dataset.vms[vm_id].execution_time(self.dataset.tasks[task_id])
        task_dependencies = {dep for dep in self.state.task_dependencies}
        task_is_ready = np.array([t.is_ready for t in self.state.task_states])
        task_start_time = np.array([t.start_time for t in self.state.task_states])
        task_completion_time = np.array([t.completion_time for t in self.state.task_states])
        vm_completion_time = np.array([v.completion_time for v in self.state.vm_states])
        task_assigned_vm_id = np.array(
            [-1 if t.assigned_vm_id is None else t.assigned_vm_id for t in self.state.task_states]
        )
        vm_assigned_task_id = np.array(
            [-1 if v.assigned_task_id is None else v.assigned_task_id for v in self.state.vm_states]
        )

        done = _assign_vm(
            task_id,
            vm_id,
            processing_time,
            task_dependencies,
            task_is_ready,
            task_start_time,
            task_completion_time,
            vm_completion_time,
            task_assigned_vm_id,
            vm_assigned_task_id,
        )

        # Convert back to objects
        new_task_states = [
            TaskState(
                is_ready=task_is_ready[t_id],
                start_time=task_start_time[t_id],
                completion_time=task_completion_time[t_id],
                assigned_vm_id=None if task_assigned_vm_id[t_id] == -1 else task_assigned_vm_id[t_id],
            )
            for t_id in range(len(self.state.task_states))
        ]
        new_vm_states = [
            VmState(
                completion_time=vm_completion_time[v_id],
                assigned_task_id=None if vm_assigned_task_id[v_id] == -1 else vm_assigned_task_id[v_id],
            )
            for v_id in range(len(self.state.vm_states))
        ]

        self.state = SimulationState(new_task_states, new_vm_states, task_dependencies)
        return None, done

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def to_assignments(self) -> list[VmAssignment]:
        assignments: list[tuple[float, VmAssignment]] = []
        for task_id, task_state in enumerate(self.state.task_states):
            if task_state.assigned_vm_id is None:
                continue  # No VM Assigned
            assignment = VmAssignment(
                task_id=task_id,
                vm_id=task_state.assigned_vm_id,
                start_time=task_state.start_time,
            )
            assignments.append((task_state.completion_time, assignment))

        assignments.sort(key=lambda x: x[0])
        return [assignment[1] for assignment in assignments]

    def makespan(self) -> float:
        return max(
            task_completion_time_est(
                self.dataset, self.state.task_states, self.state.vm_states, self.state.task_dependencies
            )
        )

    def total_energy_consumption(self) -> float:
        return sum(task_energy_consumption_est(self.dataset, self.state.task_states))

    def total_latency_score(self) -> float:
        return sum(
            task_latency_score_est(
                self.dataset, self.state.task_states, self.state.vm_states, self.state.task_dependencies
            )
        )


def _assign_vm(
    task_id: int,
    vm_id: int,
    processing_time: float,
    task_dependencies: set[tuple[int, int]],
    task_is_ready: np.ndarray,
    task_start_time: np.ndarray,
    task_completion_time: np.ndarray,
    vm_completion_time: np.ndarray,
    task_assigned_vm_id: np.ndarray,
    vm_assigned_task_id: np.ndarray,
) -> bool:
    child_task_ids = [c_id for (p_id, c_id) in task_dependencies if p_id == task_id]
    parent_task_ids = [p_id for (p_id, c_id) in task_dependencies if c_id == task_id]

    # Original values we need
    vm_prev_task_id = vm_assigned_task_id[vm_id]

    # Update scheduled states
    task_assigned_vm_id[task_id] = vm_id
    vm_assigned_task_id[vm_id] = task_id

    # Update ready states using new state
    task_is_ready[task_id] = False
    for child_id in child_task_ids:
        child_parent_task_ids = [p_id for (p_id, c_id) in task_dependencies if c_id == child_id]
        task_is_ready[child_id] = (task_assigned_vm_id[child_parent_task_ids] != -1).all()

    # Update completion times
    start_time = task_completion_time[parent_task_ids].max(initial=vm_completion_time[vm_id])
    task_start_time[task_id] = start_time
    task_completion_time[task_id] = start_time + processing_time
    vm_completion_time[vm_id] = start_time + processing_time

    # New dependencies (a new edge between the old task in the VM and this task)
    if vm_prev_task_id != -1:
        task_dependencies.add((vm_prev_task_id, task_id))

    # Find whether there are any more tasks remaining
    done: bool = (task_assigned_vm_id != -1).all()

    return done
