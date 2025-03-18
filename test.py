import random
import time

import numpy as np
import torch
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
from algorithms.moheft import MoheftScheduler
from algorithms.random import RandomScheduler
from algorithms.round_robin import RoundRobinScheduler
from constants import (
    ENERGY_CONSUMPTION_PREFERENCE,
    EVALUATION_SEED,
    LATENCY_SCORE_PREFERENCE,
    MAKESPAN_PREFERENCE,
    N_HOST,
    N_TASK,
    N_VM,
    N_WORKFLOW_TASK,
)
from dataset.generator import generate_dataset
from dataset.models import Dataset, Solution
from dataset.synthetic import SyntheticDatasetArgs
from visualizers.summary_chart import plot_summary_charts


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
        MoheftScheduler(solution_count=7, selected_solution=0),
        DrlAgentScheduler("Proposed", model_path="logs/1742091521_gnn_[1][1][1]/model.pt", agent_type="gnn"),
    ]

    table = ProgressTable(print_header_every_n_rows=0, pbar_embedded=False, pbar_show_eta=True)
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
            table.update("latency_score", value=solution.latency_score())
            table.update("runtime", value=run_end_time - run_start_time)
            table.next_row()
            progress_bar.update(1)

        sch_start_i = sch_i * (len(datasets) + 1)
        sch_end_i = sch_start_i + len(datasets)
        sch_makespan = table.at[sch_start_i:sch_end_i, 2]
        sch_e_consumption = table.at[sch_start_i:sch_end_i, 3]
        sch_latency = table.at[sch_start_i:sch_end_i, 4]
        sch_runtime = table.at[sch_start_i:sch_end_i, 5]

        avg_makespan = sum(sch_makespan) / len(sch_makespan)
        avg_energy = sum(sch_e_consumption) / len(sch_e_consumption)
        avg_latency = sum(sch_latency) / len(sch_latency)
        avg_runtime = sum(sch_runtime) / len(sch_runtime)
        summary_data.append((scheduler.name, avg_makespan, avg_energy, avg_latency, avg_runtime))

        table.update("name", scheduler.name)
        table.update("makespan", value=avg_makespan, cell_color="bold")
        table.update("energy_consumption", value=avg_energy, cell_color="bold")
        table.update("latency_score", value=avg_latency, cell_color="bold")
        table.update("runtime", value=avg_runtime, cell_color="bold")
        table.next_row(split=True)

        # _, axes = plt.subplots(nrows=1, ncols=2)
        # plot_workflow_graphs(axes[0], solution)
        # plot_gantt_chart(axes[1], solution)
        # plt.title(scheduler.name)
        # plt.show()

    table.close()

    print("\nSummary:")
    summary_table = ProgressTable(print_header_every_n_rows=0)
    for row in summary_data:
        summary_table.update("name", row[0], width=15)
        summary_table.update("makespan", row[1], width=20)
        summary_table.update("energy_consumption", row[2], width=20)
        summary_table.update("latency", row[3], width=10)
        summary_table.update("runtime", row[4], width=10)
        summary_table.next_row()
    higlight_best_results(summary_table, 1)
    higlight_best_results(summary_table, 2)
    higlight_best_results(summary_table, 3)
    higlight_best_results(summary_table, 4)
    summary_table.close()

    plot_summary_charts(table.to_df())


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    rng = np.random.RandomState(EVALUATION_SEED)
    datasets = [
        generate_dataset(
            rng=rng,
            args=SyntheticDatasetArgs(
                task_count=N_TASK,
                max_vm_count=N_VM,
                max_host_count=N_HOST,
                max_tasks_per_workflow=N_WORKFLOW_TASK,
                makespan_preference=MAKESPAN_PREFERENCE,
                energy_consumption_preference=ENERGY_CONSUMPTION_PREFERENCE,
                latency_score_preference=LATENCY_SCORE_PREFERENCE,
            ),
        )
        for _ in range(4)
    ]
    run_evaluation(datasets)
