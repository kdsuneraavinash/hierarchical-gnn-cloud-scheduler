from typing import Any
import numpy as np

from dataset.models import Dataset, Host, Task, Vm


def random_list(arr_len: int, arr_sum: int, rng: np.random.RandomState, min_value: int = 1) -> list[int]:
    value_arr = rng.rand(arr_len)
    value_arr: np.ndarray[tuple[int, ...], Any] = value_arr / value_arr.sum()
    scaling_factor = arr_sum - (arr_len * min_value)
    value_arr = np.floor(value_arr * scaling_factor) + min_value
    value_arr[-1] += arr_sum - value_arr.sum()
    return value_arr.astype(int).tolist()


def safe_clone(sub_index: int, dataset: Dataset) -> Dataset:
    """Make sure actual values are not carried over"""

    return Dataset(
        key=f"{dataset.key}-{sub_index}",
        preference=dataset.preference,
        workflows=dataset.workflows,
        tasks=[
            Task(
                id=task.id,
                workflow_id=task.workflow_id,
                length=task.length,
                child_ids=task.child_ids,
                req_memory_gb=task.req_memory_gb,
                req_disk_gb=task.req_disk_gb,
                actual_length=task.length,
            )
            for task in dataset.tasks
        ],
        vms=[
            Vm(
                id=vm.id,
                host_id=vm.host_id,
                cpu_speed_mips=vm.cpu_speed_mips,
                memory_gb=vm.memory_gb,
                disk_gb=vm.disk_gb,
                actual_cpu_speed_mips=vm.cpu_speed_mips,
            )
            for vm in dataset.vms
        ],
        hosts=[
            Host(
                id=host.id,
                cores=host.cores,
                cpu_speed_mips=host.cpu_speed_mips,
                power_idle_watt=host.power_idle_watt,
                power_peak_watt=host.power_peak_watt,
                actual_cpu_speed_mips=host.cpu_speed_mips,
                actual_power_idle_watt=host.power_idle_watt,
                actual_power_peak_watt=host.power_peak_watt,
            )
            for host in dataset.hosts
        ],
        vm_events=[],
    )
