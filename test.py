import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from progress_table import ProgressTable
from progress_table.v1.progress_table import TableProgressBar

from algorithms.base_abstract import BaseAbstractScheduler
from algorithms.drl_agent import DrlAgentScheduler
from algorithms.energy_aware import EnergyAwareSchduler
from algorithms.ferpts import FerptsScheduler
from algorithms.heft import HeftScheduler
from algorithms.least_loaded_first import LeastLoadedFirstScheduler
from algorithms.max_min import MaxMinScheduler
from algorithms.min_min import MinMinScheduler
from algorithms.random import RandomScheduler
from algorithms.round_robin import RoundRobinScheduler
from algorithms.sla_aware import SlaAwareSchduler
from constants import INT_INFINITY
from dataset.generator import DatasetArgs, generate_dataset
from dataset.models import Dataset, Solution
from visualizers.summary_chart import plot_summary_chart


def higlight_best_results(table: ProgressTable, col_index: int) -> None:
    sorted_indices = np.array([x[col_index] for x in table.to_list() if x[col_index] is not None]).argsort()
    table.at[int(sorted_indices[0]), col_index, "C"] = "bold yellow"
    table.at[int(sorted_indices[1]), col_index, "C"] = "yellow"


def run_evaluation(datasets: list[Dataset]) -> None:
    schedulers: list[BaseAbstractScheduler] = [
        HeftScheduler(),
        FerptsScheduler(),
        RandomScheduler(),
        LeastLoadedFirstScheduler(),
        MinMinScheduler(),
        MaxMinScheduler(),
        RoundRobinScheduler(),
        EnergyAwareSchduler(alpha=0.5),
        SlaAwareSchduler(alpha=0.5),
        DrlAgentScheduler(name="GIN", model_path="logs/1741699077_gin/model.pt", agent_type="gin"),
    ]

    table = ProgressTable(print_header_every_n_rows=INT_INFINITY, pbar_embedded=False, pbar_show_eta=True)
    progress_bar: TableProgressBar = table.pbar(range(len(schedulers) * len(datasets)))
    summary_data: list[tuple[str, float, float, float, float]] = []

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
            table.update("sla_penalty", value=solution.sla_penalty())
            table.update("runtime", value=run_end_time - run_start_time)
            table.next_row()
            progress_bar.update(1)

        sch_start_i = sch_i * (len(datasets) + 1)
        sch_end_i = sch_start_i + len(datasets)
        sch_makespan = table.at[sch_start_i:sch_end_i, 2]
        sch_e_consumption = table.at[sch_start_i:sch_end_i, 3]
        sch_sla_penalty = table.at[sch_start_i:sch_end_i, 4]
        sch_runtime = table.at[sch_start_i:sch_end_i, 5]

        avg_makespan = sum(sch_makespan) / len(sch_makespan)
        avg_energy = sum(sch_e_consumption) / len(sch_e_consumption)
        avg_sla_penalty = sum(sch_sla_penalty) / len(sch_sla_penalty)
        avg_runtime = sum(sch_runtime) / len(sch_runtime)
        summary_data.append((scheduler.name, avg_makespan, avg_energy, avg_sla_penalty, avg_runtime))

        table.update("name", scheduler.name)
        table.update("makespan", value=avg_makespan, cell_color="bold")
        table.update("energy_consumption", value=avg_energy, cell_color="bold")
        table.update("sla_penalty", value=avg_sla_penalty, cell_color="bold")
        table.update("runtime", value=avg_runtime, cell_color="bold")
        table.next_row(split=True)

        # _, axes = plt.subplots(nrows=1, ncols=2)
        # plot_workflow_graphs(axes[0], solution)
        # plot_gantt_chart(axes[1], solution)
        # plt.title(scheduler.name)
        # plt.show()

    table.close()

    print("\nSummary:")
    summary_table = ProgressTable(print_header_every_n_rows=INT_INFINITY)
    for row in summary_data:
        summary_table.update("name", row[0], width=15)
        summary_table.update("makespan", row[1], width=20)
        summary_table.update("energy_consumption", row[2], width=20)
        summary_table.update("sla_penalty", row[3], width=20)
        summary_table.update("runtime", row[4], width=10)
        summary_table.next_row()
    higlight_best_results(summary_table, 1)
    higlight_best_results(summary_table, 2)
    higlight_best_results(summary_table, 3)
    higlight_best_results(summary_table, 4)
    summary_table.close()

    _, ax = plt.subplots(figsize=(8, 6))
    plot_summary_chart(ax, table.to_df())
    plt.show()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    datasets = [
        generate_dataset(
            seed=100_000 + i,
            args=DatasetArgs(
                task_count=200,
                max_host_count=4,
                max_vm_count=10,
                max_tasks_per_workflow=20,
            ),
        )
        for i in range(4)
    ]
    run_evaluation(datasets)
