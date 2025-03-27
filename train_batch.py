from itertools import product

import tyro
from dataset.generator import DatasetType
from models.agent import AgentType
from train import Args, train


def main(agent_type: AgentType = "gnn", dataset_type: DatasetType = "synthetic"):
    prefs = sorted(list(product([0, 1], repeat=3)), key=sum, reverse=True)
    for m, e, s in prefs:
        if m + e + s == 0:
            continue

        args = Args(
            exp_name=f"{agent_type}_{dataset_type}_[{m}][{e}][{s}]",
            agent_type=agent_type,
            dataset_type=dataset_type,
            makespan_pref=m,
            energy_pref=e,
            sla_penalty_pref=s,
            makespan_alpha=0.541,
            energy_alpha=9.093,
            sla_penalty_alpha=0.425,
        )

        print()
        print(f"--- Running experiment: {args.exp_name} ---")
        train(args)


if __name__ == "__main__":
    tyro.cli(main)
