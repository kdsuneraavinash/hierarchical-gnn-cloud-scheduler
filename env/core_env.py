from typing import Any

import gymnasium as gym
import numpy as np

from dataset.generator import DatasetArgs, generate_dataset
from env.action import EnvAction
from env.observation import EnvObs
from env.simulation import Simulation


class CoreEnvironment(gym.Env):
    prev_makespan: float = 0

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

        obs = self.to_observation()
        self.prev_makespan = obs.task_completion_time.max()
        return obs, {}

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: EnvAction) -> tuple[EnvObs, float, bool, bool, dict[str, Any]]:
        """Performs a step in the environment given an action."""
        assert self.simulation is not None, "Environment must be reset before calling step"

        error, done = self.simulation.assign_vm(action.task_id, action.vm_id)

        # Penalize invalid actions
        if error:
            penalty = sum(-1000 if task.assigned_vm_id is None else 0 for task in self.simulation.task_states)
            print(f"Error: {error}")
            return self.to_observation(), penalty, True, False, {"error": error}

        obs = self.to_observation()
        prev_makespan = self.prev_makespan
        curr_makespan = obs.task_completion_time.max()
        self.prev_makespan = curr_makespan

        # Immediate reward (neutral reward for valid intermediate steps)
        if not done:
            reward = -(curr_makespan - prev_makespan)
            return self.to_observation(), 0, False, False, {}

        reward = -curr_makespan
        info = {"assignments": self.simulation.to_assignments()}
        return obs, reward, False, True, info

    # To Observation
    # ------------------------------------------------------------------------------------------------------------------

    def to_observation(self) -> EnvObs:
        assert self.simulation is not None, "Environment must be reset before calling to_observation"

        tasks = self.simulation.dataset.tasks
        vms = self.simulation.dataset.vms
        task_states = self.simulation.task_states
        vm_states = self.simulation.vm_states

        # Task completion time - LB(O_i)
        # For tasks that are not scheduled, fill the completion time with the maximum completion time
        # max(LB(O_parent) + min(P(i, k)))
        task_completion_time = [task_state.completion_time for task_state in task_states]
        for t_id, task_state in enumerate(task_states):
            if task_state.assigned_vm_id is not None:
                task_state.completion_time = max(
                    (
                        task_states[p_id].completion_time
                        for p_id, c_id in self.simulation.task_dependencies
                        if c_id == t_id
                    ),
                    default=0,
                ) + min(tasks[t_id].length / vm.cpu_speed_mips for vm in vms if vm.is_compatible(tasks[t_id]))
        task_completion_time = np.array(task_completion_time)

        # Whether a task is already scheduled - I(O_i)
        task_state_scheduled = np.array([task_state.assigned_vm_id is not None for task_state in task_states])

        # VM completion time - T(M_k)
        vm_completion_time = np.array([vm_state.completion_time for vm_state in vm_states])

        # Whether a task is compatible with a VM - Constraint
        task_vm_compatibilities = np.array([[int(vm.is_compatible(task)) for vm in vms] for task in tasks])

        # Task-VM execution time matrix - P(i, k)
        # For incompatible task-vm combinations, fill the time cost with the average time cost of other compatible tasks
        task_vm_time_cost_unfilled = np.array([[task.length / vm.cpu_speed_mips for vm in vms] for task in tasks])
        total_time_per_task = (task_vm_compatibilities * task_vm_time_cost_unfilled).sum(axis=1)
        mean_time_per_task: np.ndarray = total_time_per_task / task_vm_compatibilities.sum(axis=1)
        mean_time_per_task = mean_time_per_task.reshape(-1, 1).repeat(len(vms), axis=1)
        task_vm_time_cost = np.where(task_vm_compatibilities == 0, mean_time_per_task, task_vm_time_cost_unfilled)

        # Whether a task is ready to be scheduled - Constraint
        task_state_ready = np.array([task_state.is_ready for task_state in task_states])

        # Task dependencies
        task_dependencies = np.array(list(self.simulation.task_dependencies)).T

        return EnvObs(
            task_completion_time=task_completion_time,
            task_state_scheduled=task_state_scheduled,
            vm_completion_time=vm_completion_time,
            task_vm_time_cost=task_vm_time_cost,
            task_vm_compatibilities=task_vm_compatibilities,
            task_state_ready=task_state_ready,
            task_dependencies=task_dependencies,
        )
