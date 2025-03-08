from dataset.models import Dataset, VmAssignment


def to_assignments(dataset: Dataset, raw_assignments: list[tuple[int, int]]) -> list[VmAssignment]:
    ordered_assignments: list[VmAssignment] = []

    raw_assignments.sort(key=lambda x: x[0])
    assert len(raw_assignments) == len(dataset.tasks), f"Unexpected number of tasks: {len(raw_assignments)}"

    for task_id, vm_id in raw_assignments:
        parent_comp_time = max(
            (
                ordered_assignments[p_id].end_time
                for p_id in range(len(ordered_assignments))
                if task_id in dataset.tasks[p_id].child_ids
            ),
            default=0,
        )
        same_vm_comp_time = max(
            (
                ordered_assignments[t_id].end_time
                for t_id in range(len(ordered_assignments))
                if ordered_assignments[t_id].vm_id == raw_assignments[t_id][1]
            ),
            default=0,
        )

        start_time = max(parent_comp_time, same_vm_comp_time)
        exe_time = dataset.tasks[task_id].length / dataset.vms[vm_id].cpu_speed_mips
        end_time = start_time + exe_time
        ordered_assignments.append(
            VmAssignment(
                task_id=task_id,
                vm_id=vm_id,
                start_time=start_time,
                end_time=end_time,
            )
        )

    ordered_assignments.sort(key=lambda x: x.end_time)
    return ordered_assignments


class BaseScheduler:
    name: str

    def __init__(self, name: str):
        self.name = name

    def schedule(self, dataset) -> list[VmAssignment]:
        raise NotImplementedError()
