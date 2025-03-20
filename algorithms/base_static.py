from abc import ABC, abstractmethod
from time import perf_counter

from algorithms.base_abstract import BaseAbstractScheduler
from dataset.models import Dataset, VmAssignment
from env.simulation import Simulation


class BaseStaticScheduler(BaseAbstractScheduler, ABC):
    last_run_time: float = 0

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        simulation = Simulation(dataset)
        start_time = perf_counter()
        assignments = self.compute_assignments(dataset)
        end_time = perf_counter()
        self.last_run_time = end_time - start_time

        for task_id, vm_id in assignments:
            error, done = simulation.assign_vm(task_id, vm_id)
            if error:
                raise ValueError(error)
        return simulation.to_assignments()

    @abstractmethod
    def compute_assignments(self, dataset: Dataset) -> list[tuple[int, int]]:
        raise NotImplementedError()

    def run_time(self):
        return self.last_run_time

    def decision_latency(self):
        return self.last_run_time
