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
    task_mask: np.ndarray[tuple[int, ...], Any]
    vm_mask: np.ndarray[tuple[int, ...], Any]
    task_dependencies: np.ndarray[tuple[int, ...], Any]


@dataclass
class EnvObsTensor:
    task_features: torch.Tensor  # (Nt, Ft)
    vm_features: torch.Tensor  # (Nt, Nv, Fv)
    task_mask: torch.Tensor  # (Nv,)
    vm_mask: torch.Tensor  # (Nv,)
    task_dependencies: torch.Tensor  # (2, Nd)


# Create
# ------------------------------------------------------------------------------------------------------------------


def create_env_obs(
    dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState], task_dependencies: set[tuple[int, int]]
) -> EnvObs:
    # For tasks that are not scheduled, fill the completion time with the maximum completion time
    # max(LB(O_parent) + min(P(i, k)))
    task_completion_time = [task_state.completion_time for task_state in task_states]
    best_vm = max(dataset.vms, key=lambda vm: vm.cpu_speed_mips)
    for t_id, task_state in enumerate(task_states):
        if task_state.assigned_vm_id is not None:
            task_completion_time[t_id] = max(
                (task_completion_time[p_id] for p_id, c_id in task_dependencies if c_id == t_id), default=0
            ) + best_vm.execution_time(dataset.tasks[t_id])

    task_features = np.array(
        [
            (
                int(task_states[t_id].assigned_vm_id is not None),
                task_completion_time[t_id],
                dataset.tasks[t_id].length,
                dataset.tasks[t_id].priority,
                dataset.preference.makespan,
                dataset.preference.energy_consumption,
                dataset.preference.sla_penalty,
            )
            for t_id in range(len(task_states))
        ],
        dtype=np.float64,
    )
    vm_features = np.array(
        [
            [
                (
                    vm_states[v_id].completion_time,
                    dataset.vms[v_id].execution_time(dataset.tasks[t_id]),
                    dataset.hosts[dataset.vms[v_id].host_id].active_power_consumption(dataset.tasks[t_id]),
                    dataset.vms[v_id].penalty(dataset.tasks[t_id]),
                    dataset.preference.makespan,
                    dataset.preference.energy_consumption,
                    dataset.preference.sla_penalty,
                )
                for v_id in range(len(vm_states))
            ]
            for t_id in range(len(task_states))
        ],
        dtype=np.float64,
    )

    assert task_features.shape[-1] == NUM_TASK_FEATURES, f"Unexpected feature count for tasks: {task_features.shape[1]}"
    assert vm_features.shape[-1] == NUM_VM_FEATURES, f"Unexpected feature count for VMs: {vm_features.shape[1]}"

    task_mask = np.array([task_state.is_ready for task_state in task_states])
    vm_mask = np.ones(len(vm_states))  # To denote failures?
    task_dependencies_arr = np.array(list(task_dependencies)).T.reshape(2, -1)

    return EnvObs(
        task_features=task_features,
        vm_features=vm_features,
        task_mask=task_mask,
        vm_mask=vm_mask,
        task_dependencies=task_dependencies_arr,
    )


# Encode
# ------------------------------------------------------------------------------------------------------------------


def encode_env_obs(obs: EnvObs) -> np.ndarray[tuple[int, ...], Any]:
    num_tasks = obs.task_features.shape[0]
    num_vms = obs.vm_features.shape[1]
    num_task_deps = obs.task_dependencies.shape[1]

    arr = np.concatenate(
        [
            np.array([num_tasks, num_vms, num_task_deps], dtype=np.int32),  # Header
            np.asarray(obs.task_features, dtype=np.float64).flatten(),  # num_tasks*NUM_TASK_FEATURES
            np.asarray(obs.vm_features, dtype=np.float64).flatten(),  # num_tasks*num_vms*NUM_VM_FEATURES
            np.asarray(obs.task_mask, dtype=np.int32),  # num_tasks
            np.asarray(obs.vm_mask, dtype=np.int32),  # num_vms
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

    task_features = tensor[: num_tasks * NUM_TASK_FEATURES].reshape(num_tasks, -1)
    tensor = tensor[num_tasks * NUM_TASK_FEATURES :]
    vm_features = tensor[: num_tasks * num_vms * NUM_VM_FEATURES].reshape(num_tasks, num_vms, -1)
    tensor = tensor[num_tasks * num_vms * NUM_VM_FEATURES :]
    task_mask = tensor[:num_tasks].long()
    tensor = tensor[num_tasks:]
    vm_mask = tensor[:num_vms].long()
    tensor = tensor[num_vms:]
    task_dependencies = tensor[: num_task_deps * 2].reshape(2, num_task_deps).long()
    tensor = tensor[num_task_deps * 2 :]

    assert not tensor.any(), "There are non-zero elements in the padding"

    return EnvObsTensor(
        task_features=task_features,
        vm_features=vm_features,
        task_mask=task_mask,
        vm_mask=vm_mask,
        task_dependencies=task_dependencies,
    )
