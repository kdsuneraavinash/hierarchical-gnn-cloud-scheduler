import random

import numpy as np
import torch

from algorithms.random import RandomScheduler
from constants import N_VM
from dataset.generator import DatasetArgs
from env.gym_env import GymEnvironment


def main() -> None:
    n_samples = 50

    scheduler = RandomScheduler()
    dataset_args = DatasetArgs.create("synthetic")
    makespan_mean = -1
    for m, e, s in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        dataset_args.makespan_preference = m
        dataset_args.energy_consumption_preference = e
        dataset_args.sla_penalty_preference = s
        env = GymEnvironment(dataset_args=dataset_args)

        total_returns = []
        for seed in range(n_samples):
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            env.reset(seed=seed)
            curr_episodic_return = 0
            while True:
                task_id, vm_id = scheduler.select_task_and_vm(env.simulation.dataset, env.simulation.state)
                action = task_id * N_VM + vm_id
                _, reward, terminated, truncated, _ = env.step(action)
                curr_episodic_return += reward
                if terminated or truncated:
                    break
            total_returns.append(curr_episodic_return)

        mean = np.mean(total_returns)
        if makespan_mean == -1:
            makespan_mean = mean
            print(1)
        else:
            print(makespan_mean / mean)


if __name__ == "__main__":
    main()
