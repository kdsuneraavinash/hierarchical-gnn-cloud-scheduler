import random
import numpy as np
import torch
from typing import List

import tyro

from algorithms.base_dynamic import BaseDynamicScheduler
from algorithms.random import RandomScheduler
from constants import N_VM
from dataset.generator import DatasetArgs
from env.gym_env import GymEnvironment


def evaluate_scheduler(scheduler: BaseDynamicScheduler, dataset_args: DatasetArgs, n_samples: int = 50) -> float:
    env = GymEnvironment(dataset_args=dataset_args)

    returns: List[float] = []
    for seed in range(n_samples):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env.reset(seed=seed)
        total_reward = 0
        done = False
        while not done:
            assert env.simulation is not None
            task_id, vm_id = scheduler.select_task_and_vm(env.simulation.dataset, env.simulation.state)
            action = task_id * N_VM + vm_id
            _, reward, terminated, truncated, _ = env.step(np.int64(action))
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)

    return float(np.mean(returns))


def main(n_samples: int = 50) -> None:
    scheduler = RandomScheduler()
    dataset_args = DatasetArgs.create("synthetic")

    preference_sets = {
        "makespan": [1, 0, 0],
        "energy": [0, 1, 0],
        "sla": [0, 0, 1],
    }

    for name, (m, e, s) in preference_sets.items():
        dataset_args.makespan_preference = m
        dataset_args.energy_consumption_preference = e
        dataset_args.sla_penalty_preference = s

        avg_return = evaluate_scheduler(scheduler, dataset_args, n_samples=n_samples)
        ratio = -1 / avg_return if avg_return != 0 else float("inf")
        print(f"{name} alpha = {ratio:.3f}")


if __name__ == "__main__":
    tyro.cli(main)
