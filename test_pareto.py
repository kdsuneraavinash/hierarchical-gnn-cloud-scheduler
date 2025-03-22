from collections import defaultdict
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from progress_table import ProgressTable
from progress_table.v1.progress_table import TableProgressBar

from algorithms.base_mo import SolutionStoreType
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
from algorithms.base_abstract import BaseAbstractScheduler
from constants import TEST_SEED
from dataset.generator import DatasetArgs, generate_dataset
from dataset.models import Dataset, Solution
from visualizers.mo_performance import plot_mo_summary
from visualizers.pareto_front import plot_2d_pareto_fronts


moheft_store: SolutionStoreType = {}
nsga2_store: SolutionStoreType = {}
synthetic_models = [
    "logs/1742069194_gnn_syn_[0][0][1]/model.pt",
    "logs/1742072960_gnn_syn_[0][1][0]/model.pt",
    "logs/1742076667_gnn_syn_[0][1][1]/model.pt",
    "logs/1742080365_gnn_syn_[1][0][0]/model.pt",
    "logs/1742084059_gnn_syn_[1][0][1]/model.pt",
    "logs/1742087722_gnn_syn_[1][1][0]/model.pt",
    "logs/1742091521_gnn_syn_[1][1][1]/model.pt",
]
real_world_peg_models = [
    "logs/1742469300_gnn_real_peg_[1][0][0]/model.pt",
    "logs/1742507315_gnn_real_peg_[0][1][0]/model.pt",
    "logs/1742421617_gnn_real_peg_[1][1][1]/model.pt",
]
real_world_br_models = [
    "logs/1742544338_gnn_real_[1][1][0]/model.pt",
    "logs/1742535569_gnn_real_[1][1][1]/model.pt",
]
real_world_ret_models = [
    "logs/1742583459_gnn_real_branch_[1][1][1]/model.pt",
    "logs/1742616419_gnn_real_branch_[1][0][0]/model.pt",
]


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
        # *[Nsga2Scheduler(store=nsga2_store, index=i) for i in range(100)],
        *[DrlAgentScheduler("Proposed-Synthetic", model_path=model, agent_type="gnn") for model in synthetic_models],
        *[DrlAgentScheduler("Proposed-Pegasus", model_path=model, agent_type="gnn") for model in real_world_peg_models],
        *[DrlAgentScheduler("Proposed-Branch", model_path=model, agent_type="gnn") for model in real_world_br_models],
        *[DrlAgentScheduler("Proposed-Ret", model_path=model, agent_type="gnn") for model in real_world_ret_models],
    ]

    table = ProgressTable(print_header_every_n_rows=0, pbar_embedded=False, pbar_show_eta=True)
    progress_bar: TableProgressBar = table.pbar(range(len(schedulers)))
    summary_data: defaultdict[str, list[dict[str, float]]] = defaultdict(list)

    prev_scheduler_name = ""
    for scheduler in schedulers:
        if scheduler.name != prev_scheduler_name:
            table.next_row()

        table.update("name", scheduler.name)

        assignments = scheduler.schedule(dataset)

        solution = Solution(dataset, assignments)
        makespan = solution.makespan()
        energy_consumption = solution.energy_consumption()
        latency_score = solution.latency_score()
        run_time = scheduler.run_time()
        decision_latency = scheduler.decision_latency()
        table.update("makespan", value=makespan, aggregate="mean")
        table.update("energy_consumption", value=energy_consumption, aggregate="mean")
        table.update("latency_score", value=latency_score, aggregate="mean")
        table.update("run_time", value=run_time, aggregate="sum")
        table.update("decision_latency", value=decision_latency, aggregate="sum")
        progress_bar.update(1)
        prev_scheduler_name = scheduler.name
        summary_data[scheduler.name].append(
            {
                "makespan": makespan,
                "energy_consumption": energy_consumption,
                "latency_score": latency_score,
                "run_time": run_time,
                "decision_latency": decision_latency,
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
            args=DatasetArgs.real_world(),
        )
    )


if __name__ == "__main__":
    main()
