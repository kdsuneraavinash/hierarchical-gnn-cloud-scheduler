from algorithms.base_weighted_cost import BaseWeightedCostScheduler
from dataset.models import Dataset, Task, Vm
from env.state import SimulationState


class WeightedDynamicSchduler(BaseWeightedCostScheduler):
    def __init__(self) -> None:
        super().__init__("Weighted Dynamic")

    def weighted_cost(self, task: Task, vm: Vm, dataset: Dataset, state: SimulationState) -> float:
        earliest_start_time_p = max(
            (
                state.task_states[p_id].completion_time
                for p_id, p_task in enumerate(dataset.tasks)
                if task.id in p_task.child_ids
            ),
            default=0,
        )
        earliest_start_time_v = state.vm_states[vm.id].completion_time

        w_exec_time = vm.execution_time(task)
        w_power_consumption = dataset.hosts[vm.host_id].active_power_consumption(task)
        w_sla_penalty = max(earliest_start_time_v, earliest_start_time_p) * dataset.workflows[task.workflow_id].priority
        return w_exec_time + w_power_consumption + w_sla_penalty
