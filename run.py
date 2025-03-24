import random

import icecream
import numpy as np
import torch

from dataset.generator import DatasetArgs
from env.gym_env import GymEnvironment
from models.gnn_agent import GnnAgent


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    agent = GnnAgent(torch.device("cpu"))
    env = GymEnvironment(dataset_args=DatasetArgs.create("real_world"))
    obs, info = env.reset(seed=0)
    tensor_obs = torch.Tensor(obs)
    curr_iter = 0
    while True:
        action, *_ = agent.get_action_and_value(tensor_obs.reshape(1, -1))
        obs, reward, terminated, truncated, info = env.step(np.int64(action.item()))
        tensor_obs = torch.Tensor(obs)
        icecream.ic(curr_iter, action, reward)
        curr_iter += 1
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
