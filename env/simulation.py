import numpy as np
from dataset.models import Dataset, VmAssignment, VmEvent
from env.state import SimulationState, TaskState, VmState
from env.utils import task_completion_time_est, task_energy_consumption_est, task_sla_penalty_est


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
        for task in dataset.tasks:
            for child_id in task.child_ids:
                task_states[child_id].is_ready = False

        # Map to the state
        self.dataset = dataset
        self.state = SimulationState(task_states, vm_states)

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
        if not self.state.vm_states[vm_id].is_available:
            return f"{task_id=} {vm_id=}: VM not available", True

        # Convert to numpy arrays
        processing_time = self.dataset.vms[vm_id].actual_execution_time(self.dataset.tasks[task_id])
        task_dependencies = {(task.id, child_id) for task in self.dataset.tasks for child_id in task.child_ids}
        task_is_ready = np.array([t.is_ready for t in self.state.task_states])
        task_start_time = np.array([t.start_time for t in self.state.task_states])
        task_completion_time = np.array([t.completion_time for t in self.state.task_states])
        vm_completion_time = np.array([v.completion_time for v in self.state.vm_states])
        task_prev_task_id = np.array([-1 if t.prev_task_id is None else t.prev_task_id for t in self.state.task_states])
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
            task_prev_task_id,
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
                prev_task_id=None if task_prev_task_id[t_id] == -1 else task_prev_task_id[t_id],
            )
            for t_id in range(len(self.state.task_states))
        ]
        new_vm_states = [
            VmState(
                completion_time=vm_completion_time[v_id],
                assigned_task_id=None if vm_assigned_task_id[v_id] == -1 else vm_assigned_task_id[v_id],
                is_available=self.state.vm_states[v_id].is_available,
            )
            for v_id in range(len(self.state.vm_states))
        ]

        self.state = SimulationState(new_task_states, new_vm_states, time=self.state.time)
        return None, done

    def wait(self) -> bool:
        future_events = [evt for evt in self.dataset.vm_events if evt.time > self.state.time]
        event = min(future_events, key=lambda evt: evt.time, default=None)
        if event is None:
            return False

        all_task_scheduled_timed = max(task_state.start_time for task_state in self.state.task_states)
        if all_task_scheduled_timed <= event.time:
            return False

        # We will have to disregard all task assignments that happen in future
        new_task_states = [
            TaskState(is_ready=True) if task_state.start_time > event.time else task_state
            for task_state in self.state.task_states
        ]
        for task_id, task in enumerate(self.dataset.tasks):
            if new_task_states[task_id].assigned_vm_id is None:
                for child_id in task.child_ids:
                    new_task_states[child_id].is_ready = False
        # Completion time is at least the scheduled task end time
        new_vm_states = [VmState() for _ in self.dataset.vms]
        for task_id, task_state in enumerate(new_task_states):
            if task_state.assigned_vm_id is not None:
                if new_vm_states[task_state.assigned_vm_id].completion_time <= task_state.completion_time:
                    new_vm_states[task_state.assigned_vm_id].assigned_task_id = task_id
                    new_vm_states[task_state.assigned_vm_id].completion_time = task_state.completion_time
        # Completion time is at least now
        for vm_id in range(len(self.dataset.vms)):
            new_vm_states[vm_id].completion_time = max(new_vm_states[vm_id].completion_time, event.time)

        # Mark unavailability
        if event.event_type == VmEvent.T.OFF:
            new_vm_states[event.vm_id].is_available = False
        elif event.event_type == VmEvent.T.ON:
            new_vm_states[event.vm_id].is_available = True

        self.state = SimulationState(new_task_states, new_vm_states, time=event.time)
        return True

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

    # --- estimated metrics ---

    def makespan(self) -> float:
        return max(task_completion_time_est(self.dataset, self.state.task_states, self.state.vm_states))

    def total_energy_consumption(self) -> float:
        return sum(task_energy_consumption_est(self.dataset, self.state.task_states, self.state.vm_states))

    def total_sla_penalty(self) -> float:
        return sum(task_sla_penalty_est(self.dataset, self.state.task_states, self.state.vm_states))


def _assign_vm(
    task_id: int,
    vm_id: int,
    processing_time: float,
    task_dependencies: set[tuple[int, int]],
    task_is_ready: np.ndarray,
    task_start_time: np.ndarray,
    task_completion_time: np.ndarray,
    vm_completion_time: np.ndarray,
    task_prev_task_id: np.ndarray,
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
        task_prev_task_id[task_id] = vm_prev_task_id

    # Find whether there are any more tasks remaining
    done: bool = (task_assigned_vm_id != -1).all()

    return done
