from abc import ABC, abstractmethod

from dataset.models import Dataset, Task, Vm, VmAssignment
from env.simulation import Simulation
from evaluate.algorithms.base import BaseScheduler


class BaseReadyScheduler(BaseScheduler, ABC):
    _simulation: Simulation

    def __init__(self, name: str):
        super().__init__(name)

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        self._simulation = Simulation(dataset)

        done = False
        while not done:
            ready_tasks = [task for task in dataset.tasks if self._simulation.task_states[task.id].is_ready]
            task, vm = self.select_task_and_vm(ready_tasks, dataset)
            error, done = self._simulation.assign_vm(task.id, vm.id)
            if error:
                raise ValueError(error)
        return self._simulation.to_assignments()

    def select_task_and_vm(self, ready_tasks: list[Task], dataset: Dataset) -> tuple[Task, Vm]:
        next_task = self.select_task(ready_tasks, dataset)
        selected_vm = self.select_vm(next_task, dataset)
        return next_task, selected_vm

    @abstractmethod
    def select_task(self, ready_tasks: list[Task], dataset: Dataset) -> Task:
        """Out of the ready tasks, choose the next task to schedule."""
        raise NotImplementedError()

    @abstractmethod
    def select_vm(self, task: Task, dataset: Dataset) -> Vm:
        """Assign the task to a VM."""
        raise NotImplementedError()

    @property
    def task_states(self):
        return self._simulation.task_states

    @property
    def vm_states(self):
        return self._simulation.vm_states
