from matplotlib import axes, figure
import numpy as np

from constants import SCH_COLORS
from visualizers.mo_performance import find_pareto_front


def plot_pareto_front(ax: axes.Axes, data: dict[str, list[tuple[float, float]]]) -> None:
    # Plot reference pareto front from best of all points
    ref_pareto_front_results: list[tuple[float, float]] = []
    for results in data.values():
        ref_pareto_front_results.extend(results)
    ref_pareto_front_results_arr = np.array(ref_pareto_front_results)
    ref_xs, ref_ys = ref_pareto_front_results_arr[:, 0], ref_pareto_front_results_arr[:, 1]
    ref_points = np.column_stack((ref_xs, ref_ys))
    ref_pareto_indices = find_pareto_front(ref_points)
    ref_pareto_points = ref_pareto_front_results_arr[ref_pareto_indices]
    ref_pareto_sorted = ref_pareto_points[np.argsort(ref_pareto_points[:, 0])]
    ax.plot(
        ref_pareto_sorted[:, 0], ref_pareto_sorted[:, 1], marker="o", linestyle="--", label="Reference", color="black"
    )

    for scheduler, results in data.items():
        results = np.array(results)
        xs, ys = results[:, 0], results[:, 1]

        points = np.column_stack((xs, ys))
        pareto_indices = find_pareto_front(points)
        pareto_points = results[pareto_indices]

        # ax.scatter(xs, ys, label=f"_{scheduler} (Other)", alpha=0.3)
        pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], marker="o", label=scheduler, color=SCH_COLORS[scheduler])


def plot_2d_pareto_fronts(fig: figure.Figure, data: dict[str, list[dict[str, float]]], dataset_name: str) -> None:
    axes = fig.subplots(1, 3).flatten()
    metric_pairs = [
        # xlabel, ylabel, x_axis_metric_key, y_axis_metric_key
        ("Makespan", "Energy Consumption", "makespan", "energy_consumption"),
        ("QoS Penalty", "Energy Consumption", "sla_penalty", "energy_consumption"),
        ("Makespan", "QoS Penalty", "makespan", "sla_penalty"),
    ]
    for ax, (xlabel, ylabel, x_axis_metric_key, y_axis_metric_key) in zip(axes, metric_pairs):
        data_points = {k: [(w[x_axis_metric_key], w[y_axis_metric_key]) for w in v] for k, v in data.items()}
        plot_pareto_front(ax, data_points)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{dataset_name} - {xlabel}/{ylabel}")
        ax.grid()

    # Uncomment this to view legend in place of the last plot
    # fig.axes[-1].remove()
    # handles, labels = fig.axes[0].get_legend_handles_labels()
    # unique = dict(zip(labels, handles))
    # fig.legend(unique.values(), unique.keys(), loc="upper right", fontsize=12)
