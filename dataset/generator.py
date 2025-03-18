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
    arrival_rate: float = 3
    """arrival rate of workflows/second (for dynamic arrival)"""
    makespan_preference: float = 1
    """preference for optimizing makespan"""
    energy_consumption_preference: float = 1
    """preference for optimizing energy consumption"""
    latency_score_preference: float = 1
    """preference for optimizing energy consumption"""
    context: dict[str, str] = field(default_factory=dict)
    """additional context for the dataset"""

    def with_priority(self, makespan: float, energy_consumption: float, latency_score: float):
        kwargs = self.__dict__
        kwargs["makespan_preference"] = makespan
        kwargs["energy_consumption_preference"] = energy_consumption
        kwargs["latency_score_preference"] = latency_score
        return type(self)(**kwargs)


def generate_dataset(args: DatasetArgs, rng: np.random.RandomState) -> Dataset:
    from dataset.synthetic import SyntheticDatasetArgs, generate_synthetic_dataset
    from dataset.real_world import RealWorldDatasetArgs, generate_real_world_dataset

    if isinstance(args, SyntheticDatasetArgs):
        return generate_synthetic_dataset(args, rng)
    if isinstance(args, RealWorldDatasetArgs):
        return generate_real_world_dataset(args, rng)
    raise ValueError("Unknown dataset args type: " + str(args))
