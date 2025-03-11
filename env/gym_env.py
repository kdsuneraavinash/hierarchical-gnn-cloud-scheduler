from typing import Any

import gymnasium as gym
import numpy as np

from constants import INT_INFINITY, MAX_OBS_SIZE
from dataset.generator import DatasetArgs, generate_dataset
from env.observation import create_env_obs, encode_env_obs
from env.simulation import Simulation


class GymEnvironment(gym.Env[np.ndarray[tuple[int, ...], Any], np.int64]):
    prev_makespan: float = 0
    prev_energy_consumption: float = 0
    prev_sla_penalty: float = 0

    def __init__(self, dataset_args: DatasetArgs):
        super().__init__()
        self.dataset_args = dataset_args
        self.simulation: Simulation | None = None
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_OBS_SIZE,), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(INT_INFINITY, start=0)

    # Reset
    # ------------------------------------------------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray[tuple[int, ...], Any], dict[str, Any]]:
        """Resets the environment and initializes the simulation."""
        super().reset(seed=seed, options=options)

        dataset = generate_dataset(seed, self.dataset_args)
        self.simulation = Simulation(dataset)

        obs = create_env_obs(
            dataset=self.simulation.dataset,
            task_states=self.simulation.state.task_states,
            vm_states=self.simulation.state.vm_states,
            task_dependencies=self.simulation.state.task_dependencies,
        )
        self.prev_makespan = max(vm.completion_time for vm in self.simulation.state.vm_states)
        self.prev_energy_consumption = sum(task.energy_consumption for task in self.simulation.state.task_states)
        self.prev_sla_penalty = 0
        return encode_env_obs(obs), {}

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: np.int64) -> tuple[np.ndarray[tuple[int, ...], Any], float, bool, bool, dict[str, Any]]:
        """Performs a step in the environment given an action."""
        assert self.simulation is not None, "Environment must be reset before calling step"

        vm_count = len(self.simulation.dataset.vms)
        task_id = int(action // vm_count)
        vm_id = int(action % vm_count)
        error, done = self.simulation.assign_vm(task_id, vm_id)

        obs = create_env_obs(
            dataset=self.simulation.dataset,
            task_states=self.simulation.state.task_states,
            vm_states=self.simulation.state.vm_states,
            task_dependencies=self.simulation.state.task_dependencies,
        )

        # Penalize invalid actions
        if error:
            penalty = sum(-1000 if task.assigned_vm_id is None else 0 for task in self.simulation.state.task_states)
            print(f"Error: {error}")
            return encode_env_obs(obs), penalty, True, False, {"error": error}

        curr_makespan = max(vm.completion_time for vm in self.simulation.state.vm_states)
        curr_energy_consumption = sum(task.energy_consumption for task in self.simulation.state.task_states)
        new_sla_penalty = self.simulation.dataset.vms[vm_id].penalty(self.simulation.dataset.tasks[task_id])
        new_makespan = curr_makespan - self.prev_makespan

        self.prev_makespan = curr_makespan
        self.prev_energy_consumption = curr_energy_consumption
        self.prev_sla_penalty += new_sla_penalty

        if not done:
            reward = -new_makespan
            return encode_env_obs(obs), reward, False, False, {}

        reward = -self.prev_makespan
        info = {"assignments": self.simulation.to_assignments()}
        return encode_env_obs(obs), reward, False, True, info
