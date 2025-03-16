from dataclasses import dataclass, field

import numpy as np

from dataset.models import Dataset


@dataclass(kw_only=True)
class DatasetArgs:
    task_count: int
    """number of tasks"""
    max_host_count: int
    """number of hosts"""
    max_vm_count: int
    """number of VMs"""
    max_tasks_per_workflow: int
    """maximum number of tasks per workflow"""
    makespan_preference: float | None = None
    """preference for optimizing makespan"""
    energy_consumption_preference: float | None = None
    """preference for optimizing energy consumption"""
    latency_score_preference: float | None = None
    """preference for optimizing energy consumption"""
    context: dict[str, str] = field(default_factory=dict)
    """additional context for the dataset"""


def generate_dataset(args: DatasetArgs, rng: np.random.RandomState) -> Dataset:
    from dataset.synthetic import SyntheticDatasetArgs, generate_synthetic_dataset
    from dataset.real_world import RealWorldDatasetArgs, generate_real_world_dataset

    if isinstance(args, SyntheticDatasetArgs):
        return generate_synthetic_dataset(args, rng)
    if isinstance(args, RealWorldDatasetArgs):
        return generate_real_world_dataset(args, rng)
    raise ValueError("Unknown dataset args type: " + str(args))
