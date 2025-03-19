from abc import ABC, abstractmethod

from algorithms.base_abstract import BaseAbstractScheduler
from dataset.models import Dataset, VmAssignment
from env.simulation import Simulation


class BaseStaticScheduler(BaseAbstractScheduler, ABC):
    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        simulation = Simulation(dataset)
        assignments = self.compute_assignments(dataset)

        for task_id, vm_id in assignments:
            error, done = simulation.assign_vm(task_id, vm_id)
            if error:
                raise ValueError(error)
        return simulation.to_assignments()

    @abstractmethod
    def compute_assignments(self, dataset: Dataset) -> list[tuple[int, int]]:
        raise NotImplementedError()
