import random
import time

import numpy as np
import torch
from progress_table import ProgressTable
from progress_table.v1.progress_table import TableProgressBar

from dataset.generator import DatasetArgs, generate_dataset
from dataset.models import Dataset, Solution
from evaluate.algorithms.base_abstract import BaseAbstractScheduler
from evaluate.algorithms.gin_agent import GinAgentScheduler
from evaluate.algorithms.heft import HeftScheduler
from evaluate.algorithms.max_min import MaxMinScheduler
from evaluate.algorithms.min_min import MinMinScheduler
from evaluate.algorithms.random import RandomScheduler
from evaluate.algorithms.round_robin import RoundRobinScheduler


def higlight_best_results(table: ProgressTable, col_index: int):
    sorted_indices = np.array([x[col_index] for x in table.to_list() if x[col_index] is not None]).argsort()
    table.at[int(sorted_indices[0]), col_index, "C"] = "bold yellow"
    table.at[int(sorted_indices[1]), col_index, "C"] = "yellow"


def run_evaluation(datasets: list[Dataset]):
    schedulers: list[BaseAbstractScheduler] = [
        HeftScheduler(),
        RandomScheduler(),
        MinMinScheduler(),
        MaxMinScheduler(),
        RoundRobinScheduler(),
        GinAgentScheduler(name="Proposed", model_path="logs/1741426371_test/model.pt"),
    ]

    table = ProgressTable(print_header_every_n_rows=float("inf"), pbar_embedded=False, pbar_show_eta=True)
    progress_bar: TableProgressBar = table.pbar(range(len(schedulers) * len(datasets)))
    summary_data: list[tuple[str, float, float, float]] = []

    for sch_i, scheduler in enumerate(schedulers):
        table.update("name", scheduler.name, width=15)
        for d_i, dataset in enumerate(datasets):
            run_start_time = time.time()
            assignments = scheduler.schedule(dataset)
            run_end_time = time.time()

            solution = Solution(dataset, assignments)
            table.update("index", value=d_i)
            table.update("makespan", value=solution.makespan())
            table.update("energy_consumption", value=solution.energy_consumption())
            table.update("run_time", value=run_end_time - run_start_time)
            table.next_row()
            progress_bar.update(1)

        sch_start_i = sch_i * (len(datasets) + 1)
        sch_end_i = sch_start_i + len(datasets)
        sch_makespan = table.at[sch_start_i:sch_end_i, 2]
        sch_e_consumption = table.at[sch_start_i:sch_end_i, 3]
        sch_run_time = table.at[sch_start_i:sch_end_i, 4]

        avg_makespan = sum(sch_makespan) / len(sch_makespan)
        avg_energy = sum(sch_e_consumption) / len(sch_e_consumption)
        avg_runtime = sum(sch_run_time) / len(sch_run_time)
        summary_data.append((scheduler.name, avg_makespan, avg_energy, avg_runtime))

        table.update("name", scheduler.name)
        table.update("makespan", value=avg_makespan, cell_color="bold")
        table.update("energy_consumption", value=avg_energy, cell_color="bold")
        table.update("run_time", value=avg_runtime, cell_color="bold")
        table.next_row(split=True)

        # _, axes = plt.subplots(nrows=1, ncols=2)
        # plot_workflow_graphs(axes[0], solution)
        # plot_gantt_chart(axes[1], solution)
        # plt.title(scheduler.name)
        # plt.show()

    table.close()

    print("\nSummary:")
    summary_table = ProgressTable(num_decimal_places=4, print_header_every_n_rows=float("inf"))
    for row in summary_data:
        summary_table.update("name", row[0], width=15)
        summary_table.update("makespan", row[1], width=20)
        summary_table.update("energy_consumption", row[2], width=20)
        summary_table.update("run_time", row[3], width=20)
        summary_table.next_row()
    higlight_best_results(summary_table, 1)
    higlight_best_results(summary_table, 2)
    higlight_best_results(summary_table, 3)
    summary_table.close()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    datasets = [
        generate_dataset(
            DatasetArgs(
                seed=200_000 + i,
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
        )
        for i in range(10)
    ]
    run_evaluation(datasets)
