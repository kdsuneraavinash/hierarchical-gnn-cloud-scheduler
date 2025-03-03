from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np

from dataset.generator import DatasetArgs
from env.action import EnvAction
from env.observation import MAX_OBS_SIZE, EnvObs, map_env_obs
from env.core_env import CoreEnvironment


class VecEnvironment(gym.Wrapper):
    def __init__(self, env: CoreEnvironment):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_OBS_SIZE,), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(2)

    # Reset
    # ------------------------------------------------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return map_env_obs(obs), info

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: np.int32) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        assert isinstance(self.env, CoreEnvironment), "Environment must be reset before calling step"
        assert self.env.simulation is not None, "Simulation must be initialized before calling step"

        vm_count = len(self.env.simulation.dataset.vms)
        parsed_action = EnvAction(task_id=int(action // vm_count), vm_id=int(action % vm_count))

        obs, reward, done, terminal, info = super().step(parsed_action)
        return map_env_obs(obs), reward, done, terminal, info

    def makespan(self) -> float:
        assert isinstance(self.env, CoreEnvironment), "Environment must be reset before calling makespan"
        assert self.env.simulation is not None, "Simulation must be initialized before calling makespan"
        return max(vm.completion_time for vm in self.env.simulation.vm_states)

    def energy_consumption(self) -> float:
        assert isinstance(self.env, CoreEnvironment), "Environment must be reset before calling energy_consumption"
        assert self.env.simulation is not None, "Simulation must be initialized before calling energy_consumption"
        return sum(task.energy_consumption for task in self.env.simulation.task_states)
