from matplotlib import axes, figure
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

from constants import CHART_AXIS_PAD
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
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], marker="o", label=scheduler)

    ax.legend()
    ax.grid(True)


def plot_2d_pareto_fronts(fig: figure.Figure, data: dict[str, list[dict[str, float]]]) -> None:
    axes = fig.subplots(1, 2)
    metric_pairs = [
        # xlabel, ylabel, x_axis_metric_key, y_axis_metric_key
        ("Makespan", "Energy Consumption", "makespan", "energy_consumption"),
        ("Latency Score", "Energy Consumption", "latency_score", "energy_consumption"),
    ]
    for ax, (xlabel, ylabel, x_axis_metric_key, y_axis_metric_key) in zip(axes, metric_pairs):
        data_points = {k: [(w[x_axis_metric_key], w[y_axis_metric_key]) for w in v] for k, v in data.items()}
        plot_pareto_front(ax, data_points)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()


def plot_3d_pareto_front(fig: figure.Figure, data: dict[str, list[tuple[float, float, float]]]) -> None:
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
    max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")

    for scheduler, results in data.items():
        results = np.array(results)
        xs, ys, zs = results[:, 0], results[:, 1], results[:, 2]

        min_x, max_x = min(min_x, xs.min()), max(max_x, xs.max())
        min_y, max_y = min(min_y, ys.min()), max(max_y, ys.max())
        min_z, max_z = min(min_z, zs.min()), max(max_z, zs.max())

        # Find Pareto-optimal points
        points = np.column_stack((xs, ys, zs))
        pareto_indices = find_pareto_front(points)
        pareto_points = points[pareto_indices]

        # Plot all points
        ax.scatter(xs, ys, zs, label=f"_{scheduler} (Other)", alpha=0.3)  # type: ignore

        # Fit a convex hull around Pareto-optimal points (if enough points exist)
        if len(pareto_points) >= 4:
            hull = ConvexHull(pareto_points)
            for simplex in hull.simplices:
                ax.plot_trisurf(
                    pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], triangles=[simplex], alpha=0.5
                )

    ax.set_xlim(0, max_x * (1 + CHART_AXIS_PAD))
    ax.set_ylim(0, max_y * (1 + CHART_AXIS_PAD))
    ax.set_zlim(0, max_z * (1 + CHART_AXIS_PAD))
    ax.set_xlabel("Makespan")
    ax.set_ylabel("Energy Consumption")
    ax.set_zlabel("Latency Score")
    ax.legend()
