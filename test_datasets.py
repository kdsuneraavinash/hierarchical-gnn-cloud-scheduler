from typing import Literal
from dataset.synthetic import SyntheticDatasetArgs


DS_SYNTHETIC_1 = SyntheticDatasetArgs(
    task_count=50,
    max_tasks_per_workflow=10,
    min_task_length=500,
    max_task_length=100_000,
    max_vm_count=5,
    max_host_count=4,
    min_cpu_speed=500,
    max_cpu_speed=5_000,
)

DS_SYNTHETIC_2 = SyntheticDatasetArgs(
    task_count=250,
    max_tasks_per_workflow=20,
    min_task_length=250,
    max_task_length=200_000,
    max_vm_count=20,
    max_host_count=5,
    min_cpu_speed=250,
    max_cpu_speed=10_000,
)

DS_SYNTHETIC_3 = SyntheticDatasetArgs(
    task_count=1000,
    max_tasks_per_workflow=50,
    min_task_length=1,
    max_task_length=500_000,
    max_vm_count=50,
    max_host_count=10,
    min_cpu_speed=100,
    max_cpu_speed=20_000,
)

DsType = Literal["ds_syn_1", "ds_syn_2", "ds_syn_3"]


def get_dataset(ds_type: DsType):
    if ds_type == "ds_syn_1":
        return DS_SYNTHETIC_1
    if ds_type == "ds_syn_2":
        return DS_SYNTHETIC_2
    if ds_type == "ds_syn_3":
        return DS_SYNTHETIC_3
    raise ValueError("Unknown DS Type")
