from collections import defaultdict
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from progress_table import ProgressTable
from progress_table.v1.progress_table import TableProgressBar

from algorithms.base_abstract import BaseAbstractScheduler
from algorithms.drl_agent import DrlAgentScheduler
from constants import N_HOST, N_TASK, N_VM, N_WORKFLOW_TASK, TEST_SEED
from dataset.generator import DatasetArgs, generate_dataset
from dataset.models import Dataset, Solution
from visualizers.pareto_front import plot_3d_pareto_front


def run_evaluation(datasets: list[Dataset]) -> None:
    schedulers: list[BaseAbstractScheduler] = [
        DrlAgentScheduler("Proposed", model_path="logs/1741981900_gnn/model.pt", agent_type="gnn"),
    ]

    table = ProgressTable(print_header_every_n_rows=0, pbar_embedded=False, pbar_show_eta=True)
    progress_bar: TableProgressBar = table.pbar(range(len(schedulers) * len(datasets)))
    summary_data: defaultdict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for scheduler in schedulers:
        table.update("name", scheduler.name, width=15)
        for dataset in datasets:
            assignments = scheduler.schedule(dataset)

            solution = Solution(dataset, assignments)
            makespan = solution.makespan()
            energy_consumption = solution.energy_consumption()
            latency_score = solution.latency_score()
            table.update("pref_makespan", value=dataset.preference.makespan)
            table.update("pref_energy", value=dataset.preference.energy_consumption)
            table.update("pref_latency", value=dataset.preference.latency_score)
            table.update("makespan", value=makespan)
            table.update("energy_consumption", value=energy_consumption)
            table.update("latency_score", value=latency_score)
            table.next_row()
            progress_bar.update(1)
            summary_data[scheduler.name].append((makespan, energy_consumption, latency_score))

        table.next_row(split=True)

    table.close()

    fig = plt.figure(figsize=(10, 8))
    plot_3d_pareto_front(fig, summary_data)
    plt.show()


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    datasets = [
        generate_dataset(
            rng=np.random.RandomState(TEST_SEED),
            args=DatasetArgs(
                task_count=N_TASK,
                max_vm_count=N_VM,
                max_host_count=N_HOST,
                max_tasks_per_workflow=N_WORKFLOW_TASK,
                makespan_preference=m,
                energy_consumption_preference=e,
                latency_score_preference=s,
            ),
        )
        for e in range(4)
        for m in range(4)
        for s in range(4)
    ]
    run_evaluation(datasets)


if __name__ == "__main__":
    main()
