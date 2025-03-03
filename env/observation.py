from dataclasses import dataclass

import numpy as np
import torch


MAX_OBS_SIZE = 100_000


def map_env_obs(obs: "EnvObs") -> np.ndarray:
    num_tasks = obs.task_state_scheduled.shape[0]
    num_vms = obs.vm_completion_time.shape[0]
    num_task_deps = obs.task_dependencies.shape[1]
    num_compatibilities = obs.compatibilities.shape[1]

    arr = np.concatenate(
        [
            np.array([num_tasks, num_vms, num_task_deps, num_compatibilities], dtype=np.int32),  # Header
            np.asarray(obs.task_state_scheduled, dtype=np.int32),  # num_tasks
            np.asarray(obs.task_state_ready, dtype=np.int32),  # num_tasks
            np.asarray(obs.task_length, dtype=np.float64),  # num_tasks
            np.asarray(obs.vm_speed, dtype=np.float64),  # num_vms
            np.asarray(obs.vm_energy_rate, dtype=np.float64),  # num_vms
            np.asarray(obs.vm_completion_time, dtype=np.float64),  # num_vms
            np.asarray(obs.task_dependencies, dtype=np.int32).flatten(),  # num_task_deps*2
            np.asarray(obs.compatibilities, dtype=np.int32).flatten(),  # num_compatibilities*2
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
    num_compatibilities = int(tensor[3].long().item())
    tensor = tensor[4:]

    task_state_scheduled = tensor[:num_tasks].long()
    tensor = tensor[num_tasks:]
    task_state_ready = tensor[:num_tasks].long()
    tensor = tensor[num_tasks:]
    task_length = tensor[:num_tasks]
    tensor = tensor[num_tasks:]

    vm_speed = tensor[:num_vms]
    tensor = tensor[num_vms:]
    vm_energy_rate = tensor[:num_vms]
    tensor = tensor[num_vms:]
    vm_completion_time = tensor[:num_vms]
    tensor = tensor[num_vms:]

    task_dependencies = tensor[: num_task_deps * 2].reshape(2, num_task_deps).long()
    tensor = tensor[num_task_deps * 2 :]
    compatibilities = tensor[: num_compatibilities * 2].reshape(2, num_compatibilities).long()
    tensor = tensor[num_compatibilities * 2 :]

    assert not tensor.any(), "There are non-zero elements in the padding"

    return EnvObsTensor(
        task_state_scheduled=task_state_scheduled,
        task_state_ready=task_state_ready,
        task_length=task_length,
        vm_speed=vm_speed,
        vm_energy_rate=vm_energy_rate,
        vm_completion_time=vm_completion_time,
        task_dependencies=task_dependencies,
        compatibilities=compatibilities,
    )


def unmap_env_obs_batch(tensor: torch.Tensor) -> "EnvObsTensor":
    graphs = [unmap_env_obs(obs_i) for obs_i in tensor]

    task_state_scheduled = torch.concat([graph.task_state_scheduled for graph in graphs])
    task_state_ready = torch.concat([graph.task_state_ready for graph in graphs])
    task_length = torch.concat([graph.task_length for graph in graphs])
    vm_speed = torch.concat([graph.vm_speed for graph in graphs])
    vm_energy_rate = torch.concat([graph.vm_energy_rate for graph in graphs])
    vm_completion_time = torch.concat([graph.vm_completion_time for graph in graphs])

    total_task_dependencies = sum([graph.task_dependencies.shape[1] for graph in graphs])
    total_compatibilities = sum([graph.compatibilities.shape[1] for graph in graphs])
    task_dependencies = torch.zeros((2, total_task_dependencies), dtype=torch.long)
    compatibilities = torch.zeros((2, total_compatibilities), dtype=torch.long)

    # Since the task dependencies and compatibilities are in the form of edges, we need to offset the indices
    # of the second graph to avoid conflicts.
    task_offset = vm_offset = task_td_offset = cp_offset = 0
    for graph in graphs:
        num_td_edges = graph.task_dependencies.shape[1]
        num_cp_edges = graph.compatibilities.shape[1]
        task_dependencies[0][task_td_offset : task_td_offset + num_td_edges] = graph.task_dependencies[0] + task_offset
        task_dependencies[1][task_td_offset : task_td_offset + num_td_edges] = graph.task_dependencies[1] + task_offset
        compatibilities[0][cp_offset : cp_offset + num_cp_edges] = graph.compatibilities[0] + task_offset
        compatibilities[1][cp_offset : cp_offset + num_cp_edges] = graph.compatibilities[1] + vm_offset

        task_offset += graph.task_state_scheduled.shape[0]
        vm_offset += graph.vm_completion_time.shape[0]
        task_td_offset += num_td_edges
        cp_offset += num_cp_edges

    return EnvObsTensor(
        task_state_scheduled=task_state_scheduled,
        task_state_ready=task_state_ready,
        task_length=task_length,
        vm_speed=vm_speed,
        vm_energy_rate=vm_energy_rate,
        vm_completion_time=vm_completion_time,
        task_dependencies=task_dependencies,
        compatibilities=compatibilities,
    )


@dataclass
class EnvObs:
    task_state_scheduled: np.ndarray
    task_state_ready: np.ndarray
    task_length: np.ndarray
    vm_speed: np.ndarray
    vm_energy_rate: np.ndarray
    vm_completion_time: np.ndarray
    task_dependencies: np.ndarray
    compatibilities: np.ndarray


@dataclass
class EnvObsTensor:
    task_state_scheduled: torch.Tensor
    task_state_ready: torch.Tensor
    task_length: torch.Tensor
    vm_speed: torch.Tensor
    vm_energy_rate: torch.Tensor
    vm_completion_time: torch.Tensor
    task_dependencies: torch.Tensor
    compatibilities: torch.Tensor
