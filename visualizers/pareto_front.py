from typing import Any
from matplotlib import axes, figure
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


def plot_pareto_front(ax: axes.Axes, data: dict[str, list[tuple[float, float]]]) -> None:
    for scheduler, results in data.items():
        results = np.array(results)
        xs, ys = results[:, 0], results[:, 1]

        points = np.column_stack((xs, ys))
        pareto_indices = find_pareto_front(points)
        pareto_points = results[pareto_indices]

        ax.scatter(xs, ys, label=f"_{scheduler} (Other)", alpha=0.3)
        pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], marker="o", label=scheduler)

    ax.legend()
    ax.grid(True)


def plot_2d_pareto_fronts(fig: figure.Figure, data: dict[str, list[tuple[float, float, float]]]) -> None:
    axes = fig.subplots(1, 2)
    metric_pairs = [
        ("Makespan", "Energy Consumption", 0, 1),
        ("SLA Penalty", "Energy Consumption", 2, 1),
    ]
    for ax, (xlabel, ylabel, i, j) in zip(axes, metric_pairs):
        data_points = {k: [(w[i], w[j]) for w in v] for k, v in data.items()}
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

    ax.set_xlim(0, max_x * 1.1)
    ax.set_ylim(0, max_y * 1.1)
    ax.set_zlim(0, max_z * 1.1)
    ax.set_xlabel("Makespan")
    ax.set_ylabel("Energy Consumption")
    ax.set_zlabel("SLA Penalty")
    ax.legend()


def find_pareto_front(points: np.ndarray[tuple[int, ...], Any]) -> np.ndarray[tuple[int, ...], Any]:
    pareto_indices = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if all(other <= point) and any(other < point):
                # Other point is better in at least one criterion
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    return np.array(pareto_indices)
