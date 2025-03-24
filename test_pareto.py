from collections import defaultdict
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from progress_table import ProgressTable
from progress_table.progress_table import TableProgressBar

from algorithms.base_mo import SolutionStoreType
from algorithms.energy_aware import EnergyAwareSchduler
from algorithms.ferpts import FerptsScheduler
from algorithms.heft import HeftScheduler
from algorithms.least_loaded_first import LeastLoadedFirstScheduler
from algorithms.max_min import MaxMinScheduler
from algorithms.min_min import MinMinScheduler
from algorithms.moea_d import MoeaDScheduler
from algorithms.moheft import MoheftScheduler
from algorithms.nsga_2 import Nsga2Scheduler
from algorithms.nsga_3 import Nsga3Scheduler
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
nsga3_store: SolutionStoreType = {}
moead_store: SolutionStoreType = {}
synthetic_models = [
    "logs/1742069194_gnn_syn_[0][0][1]/model.pt",
    "logs/1742072960_gnn_syn_[0][1][0]/model.pt",
    "logs/1742076667_gnn_syn_[0][1][1]/model.pt",
    "logs/1742080365_gnn_syn_[1][0][0]/model.pt",
    "logs/1742084059_gnn_syn_[1][0][1]/model.pt",
    "logs/1742087722_gnn_syn_[1][1][0]/model.pt",
    "logs/1742091521_gnn_syn_[1][1][1]/model.pt",
]
real_world_real_models = [
    "logs/1742583459_gnn_real_branch_[1][1][1]/model.pt",
    "logs/1742623721_gnn_real_branch_[1][1][0]/model.pt",
    "logs/1742637987_gnn_real_branch_[0][1][1]/model.pt",
    "logs/1742616419_gnn_real_branch_[1][0][0]/model.pt",
    "logs/1742627613_gnn_real_branch_[0][1][0]/model.pt",
    "logs/1742645566_gnn_real_branch_[0][0][1]/model.pt",
    "logs/1742650703_gnn_real_branch_[1][0][1]/model.pt",
]
synthetic_mlp_models = [
    "logs/1742758554_mlp_syn_[0][0][1]/model.pt",
    "logs/1742762701_mlp_syn_[0][1][0]/model.pt",
    "logs/1742769413_mlp_syn_[0][1][1]/model.pt",
    "logs/1742771260_mlp_syn_[1][0][0]/model.pt",
    "logs/1742773104_mlp_syn_[1][0][1]/model.pt",
    "logs/1742774934_mlp_syn_[1][1][0]/model.pt",
    "logs/1742776780_mlp_syn_[1][1][1]/model.pt",
]


def run_evaluation(datasets: list[Dataset]) -> None:
    schedulers: list[BaseAbstractScheduler] = [
        # Single-Objective Schedulers
        HeftScheduler(),
        FerptsScheduler(),
        RandomScheduler(),
        LeastLoadedFirstScheduler(),
        MinMinScheduler(),
        MaxMinScheduler(),
        RoundRobinScheduler(),
        EnergyAwareSchduler(alpha=0.5),
        # Multi-Objective Schedulers
        *[MoheftScheduler(solution_count=7, store=moheft_store, index=i) for i in range(7)],
        *[Nsga2Scheduler(store=nsga2_store, index=i) for i in range(100)],
        *[Nsga3Scheduler(store=nsga3_store, index=i) for i in range(100)],
        *[MoeaDScheduler(store=moead_store, index=i) for i in range(100)],
        # *[DrlAgentScheduler("Proposed-Synthetic", model_path=model, agent_type="gnn") for model in synthetic_models],
        # *[DrlAgentScheduler("Proposed-Real", model_path=model, agent_type="gnn") for model in real_world_real_models],
        # *[DrlAgentScheduler("Proposed-MLP", model_path=model, agent_type="mlp") for model in synthetic_mlp_models],
    ]

    table = ProgressTable(print_header_every_n_rows=0, pbar_embedded=False, pbar_show_eta=True)
    progress_bar: TableProgressBar = table.pbar(range(len(schedulers) * len(datasets)))
    summary_data: defaultdict[str, list[dict[str, float]]] = defaultdict(list)

    prev_scheduler_name = schedulers[0].name
    for scheduler in schedulers:
        if scheduler.name != prev_scheduler_name:
            table.next_row()

        total_makespan: float = 0
        total_energy_consumption: float = 0
        total_sla_penalty: float = 0
        total_run_time: float = 0
        total_decision_latency: float = 0
        for dataset in datasets:
            table.update("name", scheduler.name)
            assignments = scheduler.schedule(dataset)
            solution = Solution(dataset, assignments)

            makespan = solution.actual_makespan()
            energy_consumption = solution.actual_energy_consumption()
            sla_penalty = solution.actual_sla_penalty()
            run_time = scheduler.run_time()
            decision_latency = scheduler.decision_latency()
            table.update("makespan", value=makespan, aggregate="mean")
            table.update("energy_consumption", value=energy_consumption, aggregate="mean")
            table.update("sla_penalty", value=sla_penalty, aggregate="mean")
            table.update("run_time", value=run_time, aggregate="sum")
            table.update("decision_latency", value=decision_latency, aggregate="sum")
            total_makespan += makespan
            total_energy_consumption += energy_consumption
            total_sla_penalty += sla_penalty
            total_run_time += run_time
            total_decision_latency += decision_latency
            progress_bar.update(1)

        prev_scheduler_name = scheduler.name
        summary_data[scheduler.name].append(
            {
                "makespan": total_makespan / len(datasets),
                "energy_consumption": total_energy_consumption / len(datasets),
                "sla_penalty": total_sla_penalty / len(datasets),
                "run_time": total_run_time / len(datasets),
                "decision_latency": total_decision_latency / len(datasets),
            }
        )

    table.close()

    fig = plt.figure(figsize=(12, 10))
    plot_2d_pareto_fronts(fig, summary_data)
    plt.show()

    plot_mo_summary(summary_data)


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    run_evaluation(
        [
            generate_dataset(
                dataset_key=str(key),
                rng=np.random.RandomState(TEST_SEED),
                args=DatasetArgs.synthentic(),
            )
            for key in range(1)
        ]
    )


if __name__ == "__main__":
    main()
