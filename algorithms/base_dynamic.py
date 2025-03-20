from abc import ABC, abstractmethod
from time import perf_counter

from algorithms.base_abstract import BaseAbstractScheduler
from dataset.models import Dataset, VmAssignment
from env.simulation import Simulation
from env.state import SimulationState


class BaseDynamicScheduler(BaseAbstractScheduler, ABC):
    run_time_history: list[float]

    def __init__(self, name: str):
        super().__init__(name)
        self.run_time_history = []

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        self.run_time_history.clear()
        simulation = Simulation(dataset)

        done = False
        while not done:
            start_time = perf_counter()
            task_id, vm_id = self.select_task_and_vm(dataset, simulation.state)
            end_time = perf_counter()
            self.run_time_history.append(end_time - start_time)

            error, done = simulation.assign_vm(task_id, vm_id)
            if error:
                raise ValueError(error)
        return simulation.to_assignments()

    @abstractmethod
    def select_task_and_vm(self, dataset: Dataset, state: SimulationState) -> tuple[int, int]:
        raise NotImplementedError()

    def run_time(self):
        return sum(self.run_time_history)

    def decision_latency(self):
        return sum(self.run_time_history) / len(self.run_time_history)
