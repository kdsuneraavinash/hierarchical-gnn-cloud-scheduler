from dataset.models import Dataset
from env.state import TaskState, VmState


def task_completion_time_est(dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState]) -> list[float]:
    task_completion_time = [task_state.completion_time for task_state in task_states]
    for t_id, task_state in enumerate(task_states):
        if task_state.assigned_vm_id is None:
            earliest_start_time = max(
                (task_completion_time[p_id] for p_id, p_task in enumerate(dataset.tasks) if t_id in p_task.child_ids),
                default=0,
            )
            task_completion_time[t_id] = min(
                max(vm_states[v_id].completion_time, earliest_start_time)
                + dataset.vms[v_id].execution_time(dataset.tasks[t_id])
                for v_id in range(len(vm_states))
                if dataset.vms[v_id].is_compatible(dataset.tasks[t_id], vm_states[v_id])
            )
    return task_completion_time


def task_energy_consumption_est(
    dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState]
) -> list[float]:
    task_energy_consumption: list[float] = [0 for _ in range(len(task_states))]
    for t_id, task_state in enumerate(task_states):
        if task_state.assigned_vm_id is None:
            task_energy_consumption[t_id] = min(
                dataset.hosts[vm.host_id].active_power_consumption(dataset.tasks[t_id])
                for vm in dataset.vms
                if vm.is_compatible(dataset.tasks[t_id], vm_states[vm.id])
            )
        else:
            vm = dataset.vms[task_state.assigned_vm_id]
            task_energy_consumption[t_id] = dataset.hosts[vm.host_id].active_power_consumption(dataset.tasks[t_id])
    return task_energy_consumption


def task_latency_score_est(dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState]) -> list[float]:
    task_latency_score: list[float] = []
    task_completion_time = task_completion_time_est(dataset, task_states, vm_states)
    for t_id, task_state in enumerate(task_states):
        task_latency_score_i = task_states[t_id].start_time * dataset.tasks[t_id].priority
        if task_state.assigned_vm_id is None:
            earliest_start_time_p = max(
                (task_completion_time[p_id] for p_id, p_task in enumerate(dataset.tasks) if t_id in p_task.child_ids),
                default=0,
            )
            earliest_start_time_v = max(
                vm_states[v_id].completion_time
                for v_id in range(len(vm_states))
                if dataset.vms[v_id].is_compatible(dataset.tasks[t_id], vm_states[v_id])
            )
            task_latency_score_i = max(earliest_start_time_p, earliest_start_time_v) * dataset.tasks[t_id].priority
        task_latency_score.append(task_latency_score_i)

    return task_latency_score
