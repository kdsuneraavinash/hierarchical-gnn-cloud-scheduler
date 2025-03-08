from dataclasses import dataclass

import numpy as np
import torch

from dataset.models import Dataset
from env.state import TaskState, VmState

MAX_OBS_SIZE = 100_000

# Dataclasses
# ------------------------------------------------------------------------------------------------------------------


@dataclass
class EnvObs:
    task_completion_time: np.ndarray
    task_state_scheduled: np.ndarray
    vm_completion_time: np.ndarray
    task_vm_time_cost: np.ndarray
    task_vm_compatibilities: np.ndarray
    task_state_ready: np.ndarray
    task_dependencies: np.ndarray


@dataclass
class EnvObsTensor:
    task_completion_time: torch.Tensor  # (Nt)
    task_state_scheduled: torch.Tensor  # (Nt)
    vm_completion_time: torch.Tensor  # (Nv)
    task_vm_time_cost: torch.Tensor  # (Nt, Nv)
    task_vm_compatibilities: torch.Tensor  # (Nt, Nv)
    task_state_ready: torch.Tensor  # (Nt)
    task_dependencies: torch.Tensor  # (Nd)


# Create
# ------------------------------------------------------------------------------------------------------------------


def create_env_obs(
    dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState], task_dependencies: set[tuple[int, int]]
) -> EnvObs:
    # Task completion time - LB(O_i)
    # For tasks that are not scheduled, fill the completion time with the maximum completion time
    # max(LB(O_parent) + min(P(i, k)))
    task_completion_time = [task_state.completion_time for task_state in task_states]
    for t_id, task_state in enumerate(task_states):
        if task_state.assigned_vm_id is None:
            task_completion_time[t_id] = max(
                (task_completion_time[p_id] for p_id, c_id in task_dependencies if c_id == t_id),
                default=0,
            ) + min(
                (vm.execution_time(dataset.tasks[t_id]) for vm in dataset.vms if vm.is_compatible(dataset.tasks[t_id])),
                default=float("inf"),
            )
    task_completion_time_arr = np.array(task_completion_time)

    # Whether a task is already scheduled - I(O_i)
    task_state_scheduled_arr = np.array([task_state.assigned_vm_id is not None for task_state in task_states])

    # VM completion time - T(M_k)
    vm_completion_time_arr = np.array([vm_state.completion_time for vm_state in vm_states])

    # Whether a task is compatible with a VM - Constraint
    task_vm_comp_arr = np.array([[int(vm.is_compatible(task)) for vm in dataset.vms] for task in dataset.tasks])

    # Task-VM execution time matrix - P(i, k)
    # For incompatible task-vm combinations, fill the time cost with the average time cost of other compatible tasks
    task_vm_time_cost_o = np.array([[vm.execution_time(task) for vm in dataset.vms] for task in dataset.tasks])
    total_time_per_task = (task_vm_comp_arr * task_vm_time_cost_o).sum(axis=1)
    mean_time_per_task: np.ndarray = total_time_per_task / task_vm_comp_arr.sum(axis=1)
    mean_time_per_task = mean_time_per_task.reshape(-1, 1).repeat(len(dataset.vms), axis=1)
    task_vm_time_cost_arr = np.where(task_vm_comp_arr == 0, mean_time_per_task, task_vm_time_cost_o)

    # Whether a task is ready to be scheduled - Constraint
    task_state_ready_arr = np.array([task_state.is_ready for task_state in task_states])

    # Task dependencies
    task_dependencies_arr = np.array(list(task_dependencies)).T

    return EnvObs(
        task_completion_time=task_completion_time_arr,
        task_state_scheduled=task_state_scheduled_arr,
        vm_completion_time=vm_completion_time_arr,
        task_vm_time_cost=task_vm_time_cost_arr,
        task_vm_compatibilities=task_vm_comp_arr,
        task_state_ready=task_state_ready_arr,
        task_dependencies=task_dependencies_arr,
    )


# Encode
# ------------------------------------------------------------------------------------------------------------------


def encode_env_obs(obs: EnvObs) -> np.ndarray:
    num_tasks = obs.task_state_scheduled.shape[0]
    num_vms = obs.vm_completion_time.shape[0]
    num_task_deps = obs.task_dependencies.shape[1]

    arr = np.concatenate(
        [
            np.array([num_tasks, num_vms, num_task_deps], dtype=np.int32),  # Header
            np.asarray(obs.task_completion_time, dtype=np.float64),  # num_tasks
            np.asarray(obs.task_state_scheduled, dtype=np.int32),  # num_tasks
            np.asarray(obs.vm_completion_time, dtype=np.float64),  # num_vms
            np.asarray(obs.task_vm_time_cost, dtype=np.float64).flatten(),  # num_tasks*num_vms
            np.asarray(obs.task_vm_compatibilities, dtype=np.int32).flatten(),  # num_tasks*num_vms
            np.asarray(obs.task_state_ready, dtype=np.int32),  # num_tasks
            np.asarray(obs.task_dependencies, dtype=np.int32).flatten(),  # num_task_deps*2
        ]
    )

    assert len(arr) <= MAX_OBS_SIZE, "Observation size does not fit the buffer, please adjust the size of mapper"
    arr = np.pad(arr, (0, MAX_OBS_SIZE - len(arr)), "constant")

    return arr


# Decode
# ------------------------------------------------------------------------------------------------------------------


def decode_env_obs(tensor: torch.Tensor) -> EnvObsTensor:
    assert len(tensor) == MAX_OBS_SIZE, "Tensor size is not of expected size"

    num_tasks = int(tensor[0].long().item())
    num_vms = int(tensor[1].long().item())
    num_task_deps = int(tensor[2].long().item())
    tensor = tensor[3:]

    task_completion_time = tensor[:num_tasks]
    tensor = tensor[num_tasks:]
    task_state_scheduled = tensor[:num_tasks].long()
    tensor = tensor[num_tasks:]
    vm_completion_time = tensor[:num_vms]
    tensor = tensor[num_vms:]
    task_vm_time_cost = tensor[: num_tasks * num_vms].reshape(num_tasks, num_vms)
    tensor = tensor[num_tasks * num_vms :]
    task_vm_compatibilities = tensor[: num_tasks * num_vms].reshape(num_tasks, num_vms)
    tensor = tensor[num_tasks * num_vms :]
    task_state_ready = tensor[:num_tasks].long()
    tensor = tensor[num_tasks:]
    task_dependencies = tensor[: num_task_deps * 2].reshape(2, num_task_deps).long()
    tensor = tensor[num_task_deps * 2 :]

    assert not tensor.any(), "There are non-zero elements in the padding"

    return EnvObsTensor(
        task_completion_time=task_completion_time,
        task_state_scheduled=task_state_scheduled,
        vm_completion_time=vm_completion_time,
        task_vm_time_cost=task_vm_time_cost,
        task_vm_compatibilities=task_vm_compatibilities,
        task_state_ready=task_state_ready,
        task_dependencies=task_dependencies,
    )
