from typing import Any

import gymnasium as gym
import numpy as np

from constants import ACT_SIZE, N_VM, OBS_SIZE
from dataset.generator import DatasetArgs, generate_dataset
from env.observation import (
    create_env_obs,
    encode_env_obs,
)
from env.reward import RewardFunction
from env.simulation import Simulation


class GymEnvironment(gym.Env[np.ndarray[tuple[int, ...], Any], np.int64]):
    _rng: np.random.RandomState | None = None

    def __init__(self, dataset_args: DatasetArgs):
        super().__init__()
        self.dataset_args = dataset_args
        self.simulation: Simulation | None = None
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(ACT_SIZE, start=0)
        self.reward_function = RewardFunction()

    # Reset
    # ------------------------------------------------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray[tuple[int, ...], Any], dict[str, Any]]:
        """Resets the environment and initializes the simulation."""
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        elif self._rng is None:
            self._rng = np.random.RandomState()

        dataset = generate_dataset(self.dataset_args, self._rng)
        self.simulation = Simulation(dataset)

        obs = create_env_obs(
            dataset=self.simulation.dataset,
            task_states=self.simulation.state.task_states,
            vm_states=self.simulation.state.vm_states,
        )
        self.reward_function.next_episode(self.simulation)
        return encode_env_obs(obs), {}

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: np.int64) -> tuple[np.ndarray[tuple[int, ...], Any], float, bool, bool, dict[str, Any]]:
        """Performs a step in the environment given an action."""
        assert self.simulation is not None, "Environment must be reset before calling step"

        task_id = int(action // N_VM)
        vm_id = int(action % N_VM)

        error, done = self.simulation.assign_vm(task_id, vm_id)
        obs = create_env_obs(
            dataset=self.simulation.dataset,
            task_states=self.simulation.state.task_states,
            vm_states=self.simulation.state.vm_states,
        )
        if error:
            penalty = sum(-1000 if task.assigned_vm_id is None else 0 for task in self.simulation.state.task_states)
            print(f"Error: {error}")
            return encode_env_obs(obs), penalty, True, False, {"error": error}

        reward = self.reward_function.current_reward(self.simulation, done)
        if not done:
            return encode_env_obs(obs), reward, False, False, {}

        info = {
            "assignments": self.simulation.to_assignments(),
            "makespan": self.simulation.makespan(),
            "energy_consumption": self.simulation.total_energy_consumption(),
            "latency_score": self.simulation.total_latency_score(),
        }
        return encode_env_obs(obs), reward, False, True, info
