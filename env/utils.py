from dataset.models import Dataset
from env.state import TaskState, VmState


def task_completion_time_est(dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState]) -> list[float]:
    """
    A heuristic for task completion time, every task will get a score
    which denotes the min time that task completes. As a result, the maximum
    completion time is always a min bound of the makespan.

    Tcomp_i =
        if scheduled -> Tactualcomp_i
        otherwise    -> min(
                            max(VMcomp_j, (Tcomp_k where k are parents)) + EXEC_i_j
                            where j are compatible VMs
                        )
    """
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
    """
    A heuristic of task energy consumption, every task gets the score denoting
    minimum energy consumption of that task. As a result the sum of values is
    a min bound of energy consumption.

    Tenergy_i =
        if scheduled -> Tactualenergy_i
        otherwise    -> min(ENERGY_i_j where j are compatible VMs)
    """
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


def task_sla_penalty_est(dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState]) -> list[float]:
    """
    A heuristic of SLA penalty, every task gets the score denoting
    average sla penalty of that task for the workflow. As a result the sum of values is
    a min bound of total sla penalty.

    Tsla_i = max(Tcomp_j where j is every task in the same workflow) * priority / (number of tasks in same workflow)
    """
    task_sla_penalty: list[float] = [0 for _ in range(len(task_states))]
    task_completion_time = task_completion_time_est(dataset, task_states, vm_states)
    for t_id, task_state in enumerate(task_states):
        priority = dataset.workflows[dataset.tasks[t_id].workflow_id].priority

        if task_state.assigned_vm_id is not None:
            task_sla_penalty[t_id] = task_states[t_id].start_time * priority

        elif priority != 0:
            earliest_start_time_p = max(
                (task_completion_time[p_id] for p_id, p_task in enumerate(dataset.tasks) if t_id in p_task.child_ids),
                default=0,
            )
            earliest_start_time_v = min(
                vm_states[v_id].completion_time
                for v_id in range(len(vm_states))
                if dataset.vms[v_id].is_compatible(dataset.tasks[t_id], vm_states[v_id])
            )
            task_sla_penalty[t_id] = max(earliest_start_time_p, earliest_start_time_v) * priority

    return task_sla_penalty
