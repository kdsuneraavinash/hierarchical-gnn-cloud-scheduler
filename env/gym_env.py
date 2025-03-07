from typing import Any

import gymnasium as gym
import numpy as np

from dataset.generator import DatasetArgs, generate_dataset
from env.observation import MAX_OBS_SIZE, create_env_obs, encode_env_obs
from env.simulation import Simulation


class GymEnvironment(gym.Env):
    prev_makespan: float = 0
    prev_energy_consumption: float = 0

    def __init__(self, dataset_args: DatasetArgs):
        super().__init__()
        self.dataset_args = dataset_args
        self.simulation: Simulation | None = None
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_OBS_SIZE,), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(2)

    # Reset
    # ------------------------------------------------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment and initializes the simulation."""
        super().reset(seed=seed, options=options)

        dataset_args_values = self.dataset_args.__dict__.copy()
        dataset_args_values["seed"] = seed
        dataset_args = DatasetArgs(**dataset_args_values)
        dataset = generate_dataset(dataset_args)
        self.simulation = Simulation(dataset)

        obs = create_env_obs(
            dataset=self.simulation.dataset,
            task_states=self.simulation.task_states,
            vm_states=self.simulation.vm_states,
            task_dependencies=self.simulation.task_dependencies,
        )
        self.prev_makespan = obs.task_completion_time.max()
        self.prev_energy_consumption = sum(task.energy_consumption for task in self.simulation.task_states)
        return encode_env_obs(obs), {}

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Performs a step in the environment given an action."""
        assert self.simulation is not None, "Environment must be reset before calling step"

        vm_count = len(self.simulation.dataset.vms)
        task_id = int(action // vm_count)
        vm_id = int(action % vm_count)
        error, done = self.simulation.assign_vm(task_id, vm_id)

        obs = create_env_obs(
            dataset=self.simulation.dataset,
            task_states=self.simulation.task_states,
            vm_states=self.simulation.vm_states,
            task_dependencies=self.simulation.task_dependencies,
        )

        # Penalize invalid actions
        if error:
            penalty = sum(-1000 if task.assigned_vm_id is None else 0 for task in self.simulation.task_states)
            print(f"Error: {error}")
            return encode_env_obs(obs), penalty, True, False, {"error": error}

        prev_makespan = self.prev_makespan
        curr_makespan = obs.task_completion_time.max()
        curr_energy_consumption = sum(task.energy_consumption for task in self.simulation.task_states)
        self.prev_makespan = curr_makespan
        self.prev_energy_consumption = curr_energy_consumption

        if not done:
            reward = -(curr_makespan - prev_makespan)
            return encode_env_obs(obs), reward, False, False, {}

        reward = -curr_makespan
        info = {"assignments": self.simulation.to_assignments()}
        return encode_env_obs(obs), reward, False, True, info

    def makespan(self):
        return self.prev_makespan

    def energy_consumption(self):
        return self.prev_energy_consumption
