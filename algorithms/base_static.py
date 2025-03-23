from abc import ABC, abstractmethod
from time import perf_counter

from algorithms.base_abstract import BaseAbstractScheduler
from dataset.models import Dataset, Task, Vm, VmAssignment
from env.simulation import Simulation


class BaseStaticScheduler(BaseAbstractScheduler, ABC):
    last_run_time: float = 0

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        simulation = Simulation(dataset)
        needs_rescheduling = True
        sub_index: int = 0
        while needs_rescheduling:
            # Create a dataset copy with tasks and vms disabled as necessary
            static_tasks = dataset.tasks.copy()
            static_vms = dataset.vms.copy()
            for task_id in range(len(dataset.tasks)):
                if simulation.state.task_states[task_id].assigned_vm_id is not None:
                    static_tasks[task_id] = Task(
                        id=simulation.dataset.tasks[task_id].id,
                        workflow_id=simulation.dataset.tasks[task_id].workflow_id,
                        length=0,
                        child_ids=simulation.dataset.tasks[task_id].child_ids,
                        req_memory_gb=simulation.dataset.tasks[task_id].req_memory_gb,
                        req_disk_gb=simulation.dataset.tasks[task_id].req_disk_gb,
                        priority=simulation.dataset.tasks[task_id].priority,
                    )
            for vm_id in range(len(dataset.vms)):
                if not simulation.state.vm_states[vm_id].is_available:
                    static_vms[vm_id] = Vm(
                        simulation.dataset.vms[vm_id].id,
                        host_id=simulation.dataset.vms[vm_id].host_id,
                        cpu_speed_mips=simulation.dataset.vms[vm_id].cpu_speed_mips,
                        memory_gb=-1,
                        disk_gb=-1,
                    )
            static_dataset = Dataset(
                key=f"{dataset.key}-{sub_index}",
                preference=dataset.preference,
                workflows=dataset.workflows,
                tasks=static_tasks,
                vms=static_vms,
                hosts=dataset.hosts,
                vm_events=dataset.vm_events,
            )

            start_time = perf_counter()
            assignments = self.compute_assignments(static_dataset)
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
