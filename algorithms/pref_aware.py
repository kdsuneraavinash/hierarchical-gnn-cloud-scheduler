from algorithms.base_weighted_cost import BaseWeightedCostScheduler
from dataset.models import Dataset, Task, Vm
from env.state import SimulationState


class PreferenceAwareSchduler(BaseWeightedCostScheduler):
    def __init__(self) -> None:
        super().__init__("Preference Aware")

    def weighted_cost(self, task: Task, vm: Vm, dataset: Dataset, state: SimulationState) -> float:
        return (
            dataset.preference.sla_penalty * vm.penalty(task)
            + dataset.preference.makespan * vm.execution_time(task)
            + dataset.preference.energy_consumption * dataset.hosts[vm.host_id].active_power_consumption(task)
        )
