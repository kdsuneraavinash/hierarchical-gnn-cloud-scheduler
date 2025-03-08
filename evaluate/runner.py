import time
from typing import Any

import pandas as pd

from dataset.generator import DatasetArgs, generate_dataset
from dataset.models import Dataset, Solution
from evaluate.agent import AgentScheduler
from evaluate.base import BaseScheduler
from evaluate.heft import HeftScheduler
from evaluate.random import RandomScheduler


def run_evaluation(dataset: Dataset):
    schedulers: list[BaseScheduler] = [
        RandomScheduler(),
        HeftScheduler(),
        AgentScheduler(model_path="logs/1741360518_test/model_501760.pt"),
    ]

    data: list[dict[str, Any]] = []
    for scheduler in schedulers:

        run_start_time = time.time()
        assignments = scheduler.schedule(dataset)
        run_end_time = time.time()

        solution = Solution(dataset, assignments)
        data.append(
            {
                "name": scheduler.name,
                "makespan": solution.makespan(),
                "energy_consumption": solution.energy_consumption(),
                "run_time": run_end_time - run_start_time,
            }
        )

    df = pd.DataFrame(data)
    print(df)


if __name__ == "__main__":
    dataset_args = DatasetArgs(
        seed=0,
        host_count=4,
        vm_count=10,
        workflow_count=10,
        gnp_min_n=20,
        gnp_max_n=20,
        max_memory_gb=10,
        min_cpu_speed=500,
        max_cpu_speed=5000,
        min_task_length=500,
        max_task_length=100_000,
        dag_method="gnp",
    )
    dataset = generate_dataset(dataset_args)
    run_evaluation(dataset)
