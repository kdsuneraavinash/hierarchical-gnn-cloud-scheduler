from matplotlib import axes

from dataset.models import Solution
from visualizers.utils import get_color

# Graphing functions for Gantt chart
# ----------------------------------------------------------------------------------------------------------------------


def plot_gantt_chart(ax: axes.Axes, solution: Solution, label: bool = True) -> None:
    result_map = {assignment.task_id: assignment for assignment in solution.vm_assignments}

    for task in solution.dataset.tasks:
        if task.id not in result_map:
            print(f"Not scheduled: {task=}")
            continue

        assigned_task = result_map[task.id]
        if assigned_task.start_time < 0:
            continue

        execution_time = solution.dataset.vms[assigned_task.vm_id].execution_time(task)

        ax.broken_barh(
            [(assigned_task.start_time, execution_time)],
            (int(assigned_task.vm_id) - 0.3, 0.6),
            color=get_color(task.workflow_id),
            edgecolor="black",
            linewidth=0.5,
        )
        if label:
            ax.text(
                x=assigned_task.start_time + execution_time / 2,
                y=int(assigned_task.vm_id),
                s=f"T{task.id}\n{task.length}MI\n{task.req_memory_gb // 1024}GB",
                ha="center",
                va="center",
            )

    ax.set_yticks(range(len(solution.dataset.vms)))
    ax.set_yticklabels(
        [f"VM {vm.id}\n{int(vm.cpu_speed_mips)}MIPS\n{vm.memory_gb // 1024}GB" for vm in solution.dataset.vms]
    )
    ax.set_xlabel("Time")
