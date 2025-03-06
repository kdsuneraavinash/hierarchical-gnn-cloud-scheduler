import random

import numpy as np
import torch
from icecream import ic

from dataset.generator import DatasetArgs
from env.core_env import CoreEnvironment
from env.observation import unmap_env_obs
from env.vec_env import VecEnvironment
from models.agent import GinAgent


def main():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    agent = GinAgent(torch.device("cpu"))
    env = VecEnvironment(
        CoreEnvironment(
            dataset_args=DatasetArgs(
                host_count=2,
                vm_count=2,
                workflow_count=10,
                gnp_min_n=20,
                gnp_max_n=20,
                max_memory_gb=10,
                min_cpu_speed=500,
                max_cpu_speed=5000,
                min_task_length=500,
                max_task_length=100_000,
                dag_method="gnp",
            )
        )
    )
    obs, info = env.reset(seed=0)
    tensor_obs = torch.Tensor(obs)
    ic(unmap_env_obs(tensor_obs))
    while True:
        action, *_ = agent.get_action_and_value(tensor_obs.reshape(1, -1))
        ic(action)
        obs, reward, terminated, truncated, info = env.step(action)
        tensor_obs = torch.Tensor(obs)
        ic(unmap_env_obs(tensor_obs), reward)
        if terminated or truncated:
            ic(info)
            break


if __name__ == "__main__":
    main()
