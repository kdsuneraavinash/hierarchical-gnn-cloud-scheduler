from typing import Any

import gymnasium as gym
import numpy as np

from dataset.generator import DatasetArgs, generate_dataset
from env.action import EnvAction
from env.observation import EnvObs
from env.simulation import Simulation


class CoreEnvironment(gym.Env):
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__()
        self.dataset_args = dataset_args
        self.simulation: Simulation | None = None

    # Reset
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[EnvObs, dict[str, Any]]:
        """Resets the environment and initializes the simulation."""
        super().reset(seed=seed, options=options)

        dataset_args_values = self.dataset_args.__dict__.copy()
        dataset_args_values["seed"] = seed
        dataset_args = DatasetArgs(**dataset_args_values)
        dataset = generate_dataset(dataset_args)
        self.simulation = Simulation(dataset)

        return self.to_observation(), {}

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: EnvAction) -> tuple[EnvObs, float, bool, bool, dict[str, Any]]:
        """Performs a step in the environment given an action."""
        assert self.simulation is not None, "Environment must be reset before calling step"

        error, tasks_remaining = self.simulation.assign_vm(action.task_id, action.vm_id)

        # Penalize invalid actions
        if error:
            penalty = sum(-1000 if task.assigned_vm_id is None else 0 for task in self.simulation.task_states)
            return self.to_observation(), penalty, True, False, {"error": error}

        # Immediate reward (neutral reward for valid intermediate steps)
        if tasks_remaining:
            return self.to_observation(), 0, False, False, {}

        # Terminal reward: negative of the max completion time among VM
        reward = -max(vm.completion_time for vm in self.simulation.vm_states)
        info = {"assignments": self.simulation.to_assignments()}
        return self.to_observation(), reward, False, True, info

    # To Observation
    # ------------------------------------------------------------------------------------------------------------------

    def to_observation(self) -> EnvObs:
        assert self.simulation is not None, "Environment must be reset before calling to_observation"

        tasks = self.simulation.dataset.tasks
        vms = self.simulation.dataset.vms
        hosts = self.simulation.dataset.hosts
        task_states = self.simulation.task_states
        vm_states = self.simulation.vm_states

        # Task observations
        task_state_scheduled = np.array([task_state.assigned_vm_id is not None for task_state in task_states])
        task_state_ready = np.array([task_state.is_ready for task_state in task_states])
        task_length = np.array([task.length for task in tasks])

        # VM observations
        vm_speed = np.array([vm.cpu_speed_mips for vm in vms])
        vm_energy_rate = np.array([hosts[vm.host_id].active_power_consumption_per_mi() for vm in vms])
        vm_completion_time = np.array([vm_state.completion_time for vm_state in vm_states])

        # Task-Task observations
        task_dependencies = np.array(list(self.simulation.task_dependencies)).T

        # Task-VM observations
        compatibilities_list = [(task.id, vm.id) for task in tasks for vm in vms if vm.is_compatible(task)]
        compatibilities = np.array(compatibilities_list).T

        return EnvObs(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_length=task_length,
            vm_speed=vm_speed,
            vm_energy_rate=vm_energy_rate,
            vm_completion_time=vm_completion_time,
            task_dependencies=task_dependencies,
            compatibilities=compatibilities,
        )
