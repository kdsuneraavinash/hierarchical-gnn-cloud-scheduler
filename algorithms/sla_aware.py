from algorithms.base_weighted_cost import BaseWeightedCostScheduler
from dataset.models import Dataset, Task, Vm
from env.state import SimulationState


class SlaAwareSchduler(BaseWeightedCostScheduler):
    def __init__(self, alpha: float = 1):
        super().__init__("SLA Aware")
        self.alpha = alpha

    def weighted_cost(self, task: Task, vm: Vm, dataset: Dataset, state: SimulationState) -> float:
        return self.alpha * vm.penalty(task) + (1 - self.alpha) * state.vm_states[vm.id].completion_time
