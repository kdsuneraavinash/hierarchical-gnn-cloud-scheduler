from typing import Any

import gymnasium as gym
import numpy as np

from constants import INT_INFINITY, MAX_OBS_SIZE
from dataset.generator import DatasetArgs, generate_dataset
from env.observation import (
    create_env_obs,
    encode_env_obs,
    task_completion_time_est,
    task_energy_consumption_est,
    task_sla_penalty_est,
)
from env.simulation import Simulation


class GymEnvironment(gym.Env[np.ndarray[tuple[int, ...], Any], np.int64]):
    _rng: np.random.RandomState | None = None
    _makespan_reward_buffer: list[float] = []
    _energy_consumptio_reward_buffer: list[float] = []
    _sla_penalty_reward_buffer: list[float] = []

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
        if self._rng is None:
            self._rng = np.random.RandomState(self.np_random_seed)

        dataset = generate_dataset(self.dataset_args, self._rng)
        self.simulation = Simulation(dataset)
        self._makespan_reward_buffer.clear()
        self._energy_consumptio_reward_buffer.clear()
        self._sla_penalty_reward_buffer.clear()

        obs = create_env_obs(
            dataset=self.simulation.dataset,
            task_states=self.simulation.state.task_states,
            vm_states=self.simulation.state.vm_states,
            task_dependencies=self.simulation.state.task_dependencies,
        )
        return encode_env_obs(obs), {}

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: np.int64) -> tuple[np.ndarray[tuple[int, ...], Any], float, bool, bool, dict[str, Any]]:
        """Performs a step in the environment given an action."""
        assert self.simulation is not None, "Environment must be reset before calling step"

        dataset = self.simulation.dataset
        task_states = self.simulation.state.task_states
        vm_states = self.simulation.state.vm_states
        task_dependencies = self.simulation.state.task_dependencies

        # Find the action
        vm_count = len(dataset.vms)
        task_id = int(action // vm_count)
        vm_id = int(action % vm_count)

        # Previous stats
        prev_makespan = max(task_completion_time_est(dataset, task_states, vm_states, task_dependencies))
        prev_energy_consumption = sum(task_energy_consumption_est(dataset, task_states))
        prev_sla_penalty = sum(task_sla_penalty_est(dataset, task_states))

        # Do the action
        error, done = self.simulation.assign_vm(task_id, vm_id)
        obs = create_env_obs(dataset, task_states, vm_states, task_dependencies)
        if error:
            penalty = sum(-1000 if task.assigned_vm_id is None else 0 for task in task_states)
            print(f"Error: {error}")
            return encode_env_obs(obs), penalty, True, False, {"error": error}

        # New stats
        curr_makespan = max(task_completion_time_est(dataset, task_states, vm_states, task_dependencies))
        curr_energy_consumption = sum(task_energy_consumption_est(dataset, task_states))
        curr_sla_penalty = sum(task_sla_penalty_est(dataset, task_states))

        # New delta values as reward
        makespan_reward = curr_makespan - prev_makespan
        energy_consumption_reward = curr_energy_consumption - prev_energy_consumption
        sla_penalty_reward = curr_sla_penalty - prev_sla_penalty

        # Save reward values
        self._makespan_reward_buffer.append(makespan_reward)
        self._energy_consumptio_reward_buffer.append(energy_consumption_reward)
        self._sla_penalty_reward_buffer.append(sla_penalty_reward)

        # Find normalization factor
        norm_makespan = max(float(np.mean(self._makespan_reward_buffer)), 1e-6)
        norm_energy = max(float(np.mean(self._energy_consumptio_reward_buffer)), 1e-6)
        norm_sla = max(float(np.mean(self._sla_penalty_reward_buffer)), 1e-6)

        # Final reward with preference utility
        reward = -(
            makespan_reward * dataset.preference.makespan / norm_makespan
            + energy_consumption_reward * dataset.preference.energy_consumption / norm_energy
            + sla_penalty_reward * dataset.preference.sla_penalty / norm_sla
        )

        if not done:
            return encode_env_obs(obs), reward, False, False, {}

        info = {"assignments": self.simulation.to_assignments()}
        return encode_env_obs(obs), reward, False, True, info
