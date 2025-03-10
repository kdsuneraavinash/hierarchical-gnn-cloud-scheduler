from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from constants import MAX_OBS_SIZE, NUM_TASK_FEATURES, NUM_VM_FEATURES
from dataset.models import Dataset
from env.state import TaskState, VmState

# Dataclasses
# ------------------------------------------------------------------------------------------------------------------


@dataclass
class EnvObs:
    task_features: np.ndarray[tuple[int, ...], Any]
    vm_features: np.ndarray[tuple[int, ...], Any]
    task_dependency_edges: np.ndarray[tuple[int, ...], Any]


@dataclass
class EnvObsTensor:
    task_features: torch.Tensor  # (Nt, Ft)
    vm_features: torch.Tensor  # (Nv, Fv)
    task_dependency_edges: torch.Tensor  # (2, Nd)


# Create
# ------------------------------------------------------------------------------------------------------------------


def create_env_obs(
    dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState], task_dependencies: set[tuple[int, int]]
) -> EnvObs:
    task_features = np.array(
        [
            (
                task_states[t_id].is_ready,
                int(task_states[t_id].assigned_vm_id is not None),
                task_states[t_id].completion_time,
                dataset.tasks[t_id].length,
                dataset.tasks[t_id].req_cpu_speed_mips,
                dataset.tasks[t_id].req_memory_gb,
                dataset.tasks[t_id].req_disk_gb,
                dataset.tasks[t_id].req_bandwidth_mbps,
                dataset.tasks[t_id].req_gpu,
                dataset.tasks[t_id].priority,
            )
            for t_id in range(len(task_states))
        ]
    )
    vm_features = np.array(
        [
            (
                vm_states[v_id].completion_time,
                dataset.vms[v_id].cpu_speed_mips,
                dataset.vms[v_id].memory_gb,
                dataset.vms[v_id].disk_gb,
                dataset.vms[v_id].bandwidth_mbps,
                dataset.vms[v_id].has_gpu,
            )
            for v_id in range(len(vm_states))
        ]
    )

    assert task_features.shape[1] == NUM_TASK_FEATURES, f"Unexpected feature count for tasks: {task_features.shape[1]}"
    assert vm_features.shape[1] == NUM_VM_FEATURES, f"Unexpected feature count for VMs: {vm_features.shape[1]}"

    # Task dependencies
    task_dependency_edges = np.array(list(task_dependencies)).T.reshape(2, -1)

    return EnvObs(
        task_features=task_features,
        vm_features=vm_features,
        task_dependency_edges=task_dependency_edges,
    )


# Encode
# ------------------------------------------------------------------------------------------------------------------


def encode_env_obs(obs: EnvObs) -> np.ndarray[tuple[int, ...], Any]:
    num_tasks = obs.task_features.shape[0]
    num_vms = obs.vm_features.shape[0]
    num_task_deps = obs.task_dependency_edges.shape[1]

    arr = np.concatenate(
        [
            np.array([num_tasks, num_vms, num_task_deps], dtype=np.int32),  # Header
            np.asarray(obs.task_features, dtype=np.int32).flatten(),  # num_tasks*NUM_TASK_FEATURES
            np.asarray(obs.vm_features, dtype=np.int32).flatten(),  # num_vms*NUM_VM_FEATURES
            np.asarray(obs.task_dependency_edges, dtype=np.int32).flatten(),  # num_task_deps*2
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

    task_features = tensor[: num_tasks * NUM_TASK_FEATURES].reshape(num_tasks, -1).long()
    tensor = tensor[num_tasks * NUM_TASK_FEATURES :]
    vm_features = tensor[: num_vms * NUM_VM_FEATURES].reshape(num_vms, -1).long()
    tensor = tensor[num_vms * NUM_VM_FEATURES :]
    task_dependency_edges = tensor[: num_task_deps * 2].reshape(2, num_task_deps).long()
    tensor = tensor[num_task_deps * 2 :]

    assert not tensor.any(), "There are non-zero elements in the padding"

    return EnvObsTensor(
        task_features=task_features,
        vm_features=vm_features,
        task_dependency_edges=task_dependency_edges,
    )
