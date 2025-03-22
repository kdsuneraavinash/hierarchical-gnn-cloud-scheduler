from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from constants import F_TASK, F_VM, N_TASK, N_VM, OBS_SIZE
from dataset.models import Dataset
from env.state import TaskState, VmState
from env.utils import (
    compute_task_makespan_ranks,
    task_completion_time_est,
    task_energy_consumption_est,
    task_latency_score_est,
)

# Dataclasses
# ------------------------------------------------------------------------------------------------------------------


@dataclass
class EnvObs:
    task_features: np.ndarray[tuple[int, ...], Any]  # (Nt, Ft)
    vm_features: np.ndarray[tuple[int, ...], Any]  # (Nt, Nv, Fv)
    task_mask: np.ndarray[tuple[int, ...], Any]  # (Nt,)
    vm_mask: np.ndarray[tuple[int, ...], Any]  # (Nt, Nv)
    task_dependencies: np.ndarray[tuple[int, ...], Any]  # (Nt, Nt)


@dataclass
class EnvObsTensor:
    task_features: torch.Tensor  # (B, Nt, Ft)
    vm_features: torch.Tensor  # (B, Nt, Nv, Fv)
    task_mask: torch.Tensor  # (B, Nt)
    vm_mask: torch.Tensor  # (B, Nt, Nv)
    task_dependencies: torch.Tensor  # (B, Nt, Nt)


# Create
# ------------------------------------------------------------------------------------------------------------------


def create_env_obs(dataset: Dataset, task_states: list[TaskState], vm_states: list[VmState]) -> EnvObs:
    task_makespan_ranks = compute_task_makespan_ranks(dataset)
    task_completion_time = task_completion_time_est(dataset, task_states, vm_states)
    task_energy_consumption = task_energy_consumption_est(dataset, task_states)
    task_latency_score = task_latency_score_est(dataset, task_states, vm_states)

    # --- Task Features ---

    def feat_task_is_schedulable(t_id: int) -> int:
        if t_id < len(task_states):
            return int(task_states[t_id].is_ready)
        return 0

    def feat_task_is_scheduled(t_id: int) -> int:
        if t_id < len(task_states):
            return int(task_states[t_id].assigned_vm_id is not None)
        return 0

    def feat_task_completion_time(t_id: int) -> float:
        if t_id < len(task_states):
            return task_completion_time[t_id]
        return 0

    def feat_task_makespan_rank(t_id: int) -> float:
        if t_id < len(task_states):
            return task_makespan_ranks[t_id]
        return 0

    def feat_task_energy_consumption(t_id: int) -> float:
        if t_id < len(task_states):
            return task_energy_consumption[t_id]
        return 0

    def feat_task_latency_score(t_id: int) -> float:
        if t_id < len(task_states):
            return task_latency_score[t_id]
        return 0

    def feat_task_priority(t_id: int) -> float:
        if t_id < len(task_states):
            return dataset.tasks[t_id].priority
        return 0

    # --- VM Features ---

    def feat_vm_completion_time(v_id: int) -> float:
        if v_id < len(vm_states):
            return vm_states[v_id].completion_time
        return 0

    # --- Task-VM Features ---

    def feat_task_vm_is_schedulable(t_id: int, v_id: int) -> int:
        if t_id < len(task_states) and v_id < len(vm_states):
            return int(dataset.vms[v_id].is_compatible(dataset.tasks[t_id]))
        return 0

    def feat_task_vm_execution_time(t_id: int, v_id: int) -> float:
        if t_id < len(task_states) and v_id < len(vm_states):
            return dataset.vms[v_id].execution_time(dataset.tasks[t_id])
        return 0

    def feat_task_vm_active_power_consumption(t_id: int, v_id: int) -> float:
        if t_id < len(task_states) and v_id < len(vm_states):
            return dataset.hosts[dataset.vms[v_id].host_id].active_power_consumption(dataset.tasks[t_id])
        return 0

    # --- Task-Task Features ---

    def feat_task_task_dependent(p_id: int, c_id: int) -> int:
        if p_id < len(task_states) and c_id < len(task_states):
            is_parent = c_id in dataset.tasks[p_id].child_ids
            is_vm_prev_task = task_states[c_id].prev_task_id == p_id
            return int(is_parent or is_vm_prev_task)
        return 0

    # --- Create feature vectors ---

    task_features = np.array(
        [
            (
                feat_task_is_schedulable(t_id),
                feat_task_is_scheduled(t_id),
                feat_task_completion_time(t_id),
                feat_task_makespan_rank(t_id),
                feat_task_energy_consumption(t_id),
                feat_task_latency_score(t_id),
                feat_task_priority(t_id),
            )
            for t_id in range(N_TASK)
        ],
        dtype=np.float32,
    )
    vm_features = np.array(
        [
            [
                (
                    feat_vm_completion_time(v_id),
                    feat_task_vm_is_schedulable(t_id, v_id),
                    feat_task_vm_execution_time(t_id, v_id),
                    feat_task_vm_active_power_consumption(t_id, v_id),
                )
                for v_id in range(N_VM)
            ]
            for t_id in range(N_TASK)
        ],
        dtype=np.float32,
    )

    task_mask = np.array([feat_task_is_schedulable(t_id) for t_id in range(N_TASK)])
    vm_mask = np.array(
        [[feat_task_vm_is_schedulable(t_id, v_id) for v_id in range(N_VM)] for t_id in range(N_TASK)],
        dtype=np.int32,
    )
    task_dependencies_matrix = np.array(
        [[feat_task_task_dependent(p_id, c_id) for c_id in range(N_TASK)] for p_id in range(N_TASK)],
        dtype=np.int32,
    )

    assert task_features.shape == (N_TASK, F_TASK), task_features.shape
    assert vm_features.shape == (N_TASK, N_VM, F_VM), vm_features.shape
    assert task_mask.shape == (N_TASK,), task_mask.shape
    assert vm_mask.shape == (N_TASK, N_VM), vm_mask.shape
    assert task_dependencies_matrix.shape == (N_TASK, N_TASK), task_dependencies_matrix.shape

    return EnvObs(
        task_features=task_features,
        vm_features=vm_features,
        task_mask=task_mask,
        vm_mask=vm_mask,
        task_dependencies=task_dependencies_matrix,
    )


# Encode
# ------------------------------------------------------------------------------------------------------------------


def encode_env_obs(obs: EnvObs) -> np.ndarray[tuple[int, ...], Any]:
    arr = np.concatenate(
        [
            np.asarray(obs.task_features, dtype=np.float32).flatten(),  # Nt*Ft
            np.asarray(obs.vm_features, dtype=np.float32).flatten(),  # Nt*Nv*Fv
            np.asarray(obs.task_mask, dtype=np.int32),  # Nt
            np.asarray(obs.vm_mask, dtype=np.int32).flatten(),  # Nt*Nv
            np.asarray(obs.task_dependencies, dtype=np.int32).flatten(),  # Nt*Nt
        ]
    )

    assert arr.shape == (OBS_SIZE,), arr.shape
    return arr


# Decode
# ------------------------------------------------------------------------------------------------------------------


def decode_env_obs_batched(tensor: torch.Tensor) -> EnvObsTensor:
    B = tensor.shape[0]
    assert tensor.shape == (B, OBS_SIZE), tensor.shape

    task_features_list: list[torch.Tensor] = []
    vm_features_list: list[torch.Tensor] = []
    task_mask_list: list[torch.Tensor] = []
    vm_mask_list: list[torch.Tensor] = []
    task_dependencies_list: list[torch.Tensor] = []
    for tensor_i in tensor:
        task_features_list.append(tensor_i[: N_TASK * F_TASK].reshape(N_TASK, F_TASK))
        offset = N_TASK * F_TASK
        vm_features_list.append(tensor_i[offset : offset + N_TASK * N_VM * F_VM].reshape(N_TASK, N_VM, F_VM))
        offset += N_TASK * N_VM * F_VM
        task_mask_list.append(tensor_i[offset : offset + N_TASK].long())
        offset += N_TASK
        vm_mask_list.append(tensor_i[offset : offset + N_TASK * N_VM].reshape(N_TASK, N_VM).long())
        offset += N_TASK * N_VM
        task_dependencies_list.append(tensor_i[offset : offset + N_TASK * N_TASK].reshape(N_TASK, N_TASK).long())

    task_features = torch.stack(task_features_list)
    vm_features = torch.stack(vm_features_list)
    task_mask = torch.stack(task_mask_list)
    vm_mask = torch.stack(vm_mask_list)
    task_dependencies = torch.stack(task_dependencies_list)

    return EnvObsTensor(
        task_features=task_features,
        vm_features=vm_features,
        task_mask=task_mask,
        vm_mask=vm_mask,
        task_dependencies=task_dependencies,
    )
