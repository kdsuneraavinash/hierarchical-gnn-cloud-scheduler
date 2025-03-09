from abc import ABC, abstractmethod

from dataset.models import Dataset, VmAssignment
from env.simulation import Simulation, SimulationState
from evaluate.algorithms.base_abstract import BaseAbstractScheduler


class BaseDynamicScheduler(BaseAbstractScheduler, ABC):
    def __init__(self, name: str):
        super().__init__(name)

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        simulation = Simulation(dataset)

        done = False
        while not done:
            task_id, vm_id = self.select_task_and_vm(dataset, simulation.state)
            error, done = simulation.assign_vm(task_id, vm_id)
            if error:
                raise ValueError(error)
        return simulation.to_assignments()

    @abstractmethod
    def select_task_and_vm(self, dataset: Dataset, state: SimulationState) -> tuple[int, int]:
        raise NotImplementedError()
