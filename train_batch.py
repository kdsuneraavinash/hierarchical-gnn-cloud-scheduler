from itertools import product

import tyro
from dataset.generator import DatasetType
from models.agent import AgentType
from train import Args, train


def main(agent_type: AgentType = "mlp", dataset_type: DatasetType = "synthetic"):
    prefs = sorted(list(product([0, 1], repeat=3)), key=sum, reverse=True)
    completed_prefs = set()
    for m, e, s in prefs:
        if m + e + s == 0:
            continue

        max_pref = max(m, e, s)
        m, e, s = m / max_pref, e / max_pref, s / max_pref
        if (m, e, s) in completed_prefs:
            continue
        completed_prefs.add((m, e, s))

        args = Args(
            exp_name=f"{agent_type}_{dataset_type}_[{m}][{e}][{s}]",
            agent_type=agent_type,
            dataset_type=dataset_type,
            makespan_pref=m,
            energy_pref=e,
            sla_penalty_pref=s,
            makespan_alpha=0.91,
            energy_alpha=14.398,
            sla_penalty_alpha=0.602,
        )

        print()
        print(f"--- Running experiment: {args.exp_name} ---")
        train(args)


if __name__ == "__main__":
    tyro.cli(main)
