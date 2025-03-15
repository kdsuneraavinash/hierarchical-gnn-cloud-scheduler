from algorithms.base_weighted_cost import BaseWeightedCostScheduler
from dataset.models import Dataset, Task, Vm
from env.state import SimulationState


class PreferenceAwareSchduler(BaseWeightedCostScheduler):
    def __init__(self) -> None:
        super().__init__("Preference Aware")

    def weighted_cost(self, task: Task, vm: Vm, dataset: Dataset, state: SimulationState) -> float:
        preference = dataset.preference
        w_exec_time = preference.makespan * vm.execution_time(task)
        w_power_consumption = preference.energy_consumption * dataset.hosts[vm.host_id].active_power_consumption(task)
        return w_exec_time + w_power_consumption
