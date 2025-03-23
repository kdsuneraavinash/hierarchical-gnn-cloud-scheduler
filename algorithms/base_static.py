from abc import ABC, abstractmethod
from time import perf_counter

from algorithms.base_abstract import BaseAbstractScheduler
from dataset.models import Dataset, VmAssignment
from dataset.utils import safe_clone
from env.simulation import Simulation


class BaseStaticScheduler(BaseAbstractScheduler, ABC):
    last_run_time: float = 0

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        simulation = Simulation(dataset)
        needs_rescheduling = True
        sub_index: int = 0
        while needs_rescheduling:
            safe_dataset = safe_clone(sub_index, dataset)
            for task_id in range(len(dataset.tasks)):
                if simulation.state.task_states[task_id].assigned_vm_id is not None:
                    safe_dataset.tasks[task_id].length = 0
            for vm_id in range(len(dataset.vms)):
                if not simulation.state.vm_states[vm_id].is_available:
                    safe_dataset.vms[vm_id].memory_gb = -1
                    safe_dataset.vms[vm_id].disk_gb = -1

            start_time = perf_counter()
            assignments = self.compute_assignments(safe_dataset)
            end_time = perf_counter()
            self.last_run_time += end_time - start_time
            sub_index += 1

            for task_id, vm_id in assignments:
                if not simulation.state.task_states[task_id].is_ready:
                    continue
                error, done = simulation.assign_vm(task_id, vm_id)
                if error:
                    raise ValueError(error)
            needs_rescheduling = simulation.wait()
        return simulation.to_assignments()

    @abstractmethod
    def compute_assignments(self, dataset: Dataset) -> list[tuple[int, int]]:
        raise NotImplementedError()

    def run_time(self):
        return self.last_run_time

    def decision_latency(self):
        return self.last_run_time
