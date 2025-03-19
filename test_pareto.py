from collections import defaultdict
import random
from time import perf_counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from progress_table import ProgressTable
from progress_table.v1.progress_table import TableProgressBar

from algorithms.base_mo import SolutionStore
from algorithms.drl_agent import DrlAgentScheduler
from algorithms.energy_aware import EnergyAwareSchduler
from algorithms.ferpts import FerptsScheduler
from algorithms.heft import HeftScheduler
from algorithms.least_loaded_first import LeastLoadedFirstScheduler
from algorithms.max_min import MaxMinScheduler
from algorithms.min_min import MinMinScheduler
from algorithms.moheft import MoheftScheduler
from algorithms.nsga_2 import Nsga2Scheduler
from algorithms.random import RandomScheduler
from algorithms.round_robin import RoundRobinScheduler
from algorithms.base_abstract import BaseAbstractScheduler
from constants import SYNTHETIC_DATASET_ARGS, TEST_SEED
from dataset.generator import generate_dataset
from dataset.models import Dataset, Solution
from visualizers.mo_performance import plot_mo_summary
from visualizers.pareto_front import plot_2d_pareto_fronts


moheft_store = SolutionStore()
nsga2_store = SolutionStore()


def run_evaluation(dataset: Dataset) -> None:
    schedulers: list[BaseAbstractScheduler] = [
        HeftScheduler(),
        FerptsScheduler(),
        RandomScheduler(),
        LeastLoadedFirstScheduler(),
        MinMinScheduler(),
        MaxMinScheduler(),
        RoundRobinScheduler(),
        EnergyAwareSchduler(alpha=0.5),
        *[MoheftScheduler(solution_count=7, store=moheft_store, index=i) for i in range(100)],
        *[Nsga2Scheduler(store=nsga2_store, index=i) for i in range(100)],
        DrlAgentScheduler("Proposed", model_path="logs/1742069194_gnn_[0][0][1]/model.pt", agent_type="gnn"),
        DrlAgentScheduler("Proposed", model_path="logs/1742072960_gnn_[0][1][0]/model.pt", agent_type="gnn"),
        DrlAgentScheduler("Proposed", model_path="logs/1742076667_gnn_[0][1][1]/model.pt", agent_type="gnn"),
        DrlAgentScheduler("Proposed", model_path="logs/1742080365_gnn_[1][0][0]/model.pt", agent_type="gnn"),
        DrlAgentScheduler("Proposed", model_path="logs/1742084059_gnn_[1][0][1]/model.pt", agent_type="gnn"),
        DrlAgentScheduler("Proposed", model_path="logs/1742087722_gnn_[1][1][0]/model.pt", agent_type="gnn"),
        DrlAgentScheduler("Proposed", model_path="logs/1742091521_gnn_[1][1][1]/model.pt", agent_type="gnn"),
    ]

    table = ProgressTable(print_header_every_n_rows=0, pbar_embedded=False, pbar_show_eta=True)
    progress_bar: TableProgressBar = table.pbar(range(len(schedulers)))
    summary_data: defaultdict[str, list[dict[str, float]]] = defaultdict(list)

    prev_scheduler_name = ""
    for scheduler in schedulers:
        if scheduler.name != prev_scheduler_name:
            table.next_row()

        table.update("name", scheduler.name)

        start_time = perf_counter()
        assignments = scheduler.schedule(dataset)
        end_time = perf_counter()

        solution = Solution(dataset, assignments)
        makespan = solution.makespan()
        energy_consumption = solution.energy_consumption()
        latency_score = solution.latency_score()
        run_time = end_time - start_time
        table.update("pref_makespan", value=dataset.preference.makespan, aggregate="mean")
        table.update("pref_energy", value=dataset.preference.energy_consumption, aggregate="mean")
        table.update("pref_latency", value=dataset.preference.latency_score, aggregate="mean")
        table.update("makespan", value=makespan, aggregate="mean")
        table.update("energy_consumption", value=energy_consumption, aggregate="mean")
        table.update("latency_score", value=latency_score, aggregate="mean")
        table.update("time", value=run_time, aggregate="sum")
        progress_bar.update(1)
        prev_scheduler_name = scheduler.name
        summary_data[scheduler.name].append(
            {
                "makespan": makespan,
                "energy_consumption": energy_consumption,
                "latency_score": latency_score,
                "run_time": run_time,
            }
        )

    table.close()

    fig = plt.figure(figsize=(12, 5))
    plot_2d_pareto_fronts(fig, summary_data)
    plt.show()

    plot_mo_summary(summary_data)


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    run_evaluation(
        generate_dataset(
            rng=np.random.RandomState(TEST_SEED),
            args=SYNTHETIC_DATASET_ARGS,
        )
    )


if __name__ == "__main__":
    main()
