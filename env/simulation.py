import copy

from dataset.models import Dataset, VmAssignment
from env.state import SimulationState, TaskState, VmState


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
        task = self.dataset.tasks[task_id]
        vm = self.dataset.vms[vm_id]

        # Checks for action
        if not (0 <= task_id < len(self.state.task_states)):
            return f"{task_id=} {vm_id=}: Invalid task (out of range)", True
        if self.state.task_states[task_id].assigned_vm_id is not None:
            return f"{task_id=} {vm_id=}: Already scheduled task", True
        if not self.state.task_states[task_id].is_ready:
            return f"{task_id=} {vm_id=}: Not ready task", True
        if not vm.is_compatible(task):
            return f"{task_id=} {vm_id=}: Task/VM are not compatible", True

        child_task_ids = [c_id for (p_id, c_id) in self.state.task_dependencies if p_id == task_id]
        parent_task_ids = [p_id for (p_id, c_id) in self.state.task_dependencies if c_id == task_id]
        processing_time = vm.execution_time(task)

        new_task_states = copy.deepcopy(self.state.task_states)
        new_vm_states = copy.deepcopy(self.state.vm_states)

        # Update scheduled states
        new_task_states[task_id].assigned_vm_id = vm_id
        new_vm_states[vm_id].assigned_task_id = task_id

        # Update ready states using new state
        new_task_states[task_id].is_ready = False
        for child_id in child_task_ids:
            new_task_states[child_id].is_ready = True
            child_parent_task_ids = [p_id for (p_id, c_id) in self.state.task_dependencies if c_id == child_id]
            for child_parent_task_id in child_parent_task_ids:
                if new_task_states[child_parent_task_id].assigned_vm_id is None:
                    new_task_states[child_id].is_ready = False
                    break

        # Update completion times
        start_time = self.state.vm_states[vm_id].completion_time
        for parent_id in parent_task_ids:
            start_time = max(start_time, self.state.task_states[parent_id].completion_time)
        new_task_states[task_id].start_time = start_time
        new_task_states[task_id].completion_time = start_time + processing_time
        new_vm_states[vm_id].completion_time = start_time + processing_time

        # Update energy consumption
        new_task_states[task_id].energy_consumption = self.dataset.hosts[vm.host_id].active_power_consumption(task)

        # New dependencies (a new edge between the old task in the VM and this task)
        new_task_dependencies = copy.deepcopy(self.state.task_dependencies)
        vm_prev_task_id = self.state.vm_states[vm_id].assigned_task_id
        if vm_prev_task_id is not None:
            new_task_dependencies.add((vm_prev_task_id, task_id))

        # Change the state
        self.state = SimulationState(new_task_states, new_vm_states, new_task_dependencies)

        # Find whether there are any more tasks remaining
        done = all(task_state.assigned_vm_id is not None for task_state in self.state.task_states)
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
