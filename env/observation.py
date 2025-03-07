from dataclasses import dataclass

import numpy as np
import torch

MAX_OBS_SIZE = 100_000


def map_env_obs(obs: "EnvObs") -> np.ndarray:
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


def unmap_env_obs(tensor: torch.Tensor) -> "EnvObsTensor":
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
