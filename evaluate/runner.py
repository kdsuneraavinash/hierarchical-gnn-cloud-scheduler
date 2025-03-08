import time
from typing import Any

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from dataset.generator import DatasetArgs, generate_dataset
from dataset.models import Dataset, Solution
from evaluate.algorithms.agent import AgentScheduler
from evaluate.algorithms.base import BaseScheduler
from evaluate.algorithms.heft import HeftScheduler
from evaluate.plotters.color import draw_agraph
from evaluate.plotters.gantt_chart import plot_gantt_chart
from evaluate.plotters.workflow_graph import plot_workflow_graphs


def run_evaluation(dataset: Dataset):
    schedulers: list[BaseScheduler] = [
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

        _, axes = plt.subplots(nrows=1, ncols=2)
        g_w: nx.DiGraph = nx.DiGraph()
        a_w = plot_workflow_graphs(g_w, solution.dataset.tasks)
        draw_agraph(axes[0], a_w)
        plot_gantt_chart(axes[1], solution)
        plt.show()

    df = pd.DataFrame(data)
    print(df)


if __name__ == "__main__":
    dataset_args = DatasetArgs(
        seed=0,
        host_count=4,
        vm_count=10,
        workflow_count=2,
        gnp_min_n=5,
        gnp_max_n=5,
        max_memory_gb=10,
        min_cpu_speed=500,
        max_cpu_speed=5000,
        min_task_length=500,
        max_task_length=100_000,
        dag_method="gnp",
    )
    dataset = generate_dataset(dataset_args)
    run_evaluation(dataset)
