from dataset.models import Dataset
from env.state import TaskState, VmState


def task_completion_time_est(
    dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState], task_dependencies: set[tuple[int, int]]
) -> list[float]:
    task_completion_time = [task_state.completion_time for task_state in task_states]
    for t_id, task_state in enumerate(task_states):
        if task_state.assigned_vm_id is None:
            earliest_start_time = max(
                (task_completion_time[p_id] for p_id, c_id in task_dependencies if c_id == t_id), default=0
            )
            task_completion_time[t_id] = min(
                max(vm_states[v_id].completion_time, earliest_start_time)
                + dataset.vms[v_id].execution_time(dataset.tasks[t_id])
                for v_id in range(len(vm_states))
            )
    return task_completion_time


def task_energy_consumption_est(dataset: Dataset, task_states: list[TaskState]) -> list[float]:
    task_energy_consumption = [task_state.energy_consumption for task_state in task_states]
    for t_id, task_state in enumerate(task_states):
        if task_state.assigned_vm_id is None:
            task_energy_consumption[t_id] = min(
                dataset.hosts[vm.host_id].active_power_consumption(dataset.tasks[t_id]) for vm in dataset.vms
            )
    return task_energy_consumption


def task_sla_penalty_est(dataset: Dataset, task_states: list[TaskState]) -> list[float]:
    sla_penalty: list[float] = [0 for _ in task_states]
    for t_id, task_state in enumerate(task_states):
        if task_state.assigned_vm_id is None:
            sla_penalty[t_id] = min(vm.penalty(dataset.tasks[t_id]) for vm in dataset.vms)
        else:
            sla_penalty[t_id] = dataset.vms[task_state.assigned_vm_id].penalty(dataset.tasks[t_id])
    return sla_penalty
