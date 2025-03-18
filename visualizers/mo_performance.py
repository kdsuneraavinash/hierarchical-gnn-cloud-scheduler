from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
import seaborn as sns


Y_PAD = 0.1


def plot_mo_summary(summary_data: dict[str, list[tuple[float, float, float]]]) -> None:
    pareto_point_map: dict[str, np.ndarray[tuple[int, ...], Any]] = {}
    for scheduler, results in summary_data.items():
        results = np.array(results)
        xs, ys = results[:, 0], results[:, 1]

        points = np.column_stack((xs, ys))
        pareto_indices = find_pareto_front(points)
        pareto_points = results[pareto_indices]
        pareto_point_map[scheduler] = pareto_points

    all_points = np.vstack([results for results in pareto_point_map.values()])
    worst_values = np.max(all_points, axis=0)
    reference_point = worst_values * 1.1

    data: list[dict[str, Any]] = []
    hv = HV(ref_point=reference_point)
    for scheduler, pareto_points in pareto_point_map.items():
        data.append({"name": scheduler, "hypervolume": hv(pareto_points), "count": len(pareto_points)})

    df = pd.DataFrame(data)
    print(df)

    df["proposed"] = df["name"] == "Proposed"
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharex=False)
    for i, metric in enumerate(["hypervolume", "count"]):
        avg_sorted = df.sort_values(metric)
        sns.barplot(data=avg_sorted, x="name", y=metric, hue="proposed", ax=axes[i], palette="Set2", legend=False)
        y_min, y_max = avg_sorted[metric].min(), avg_sorted[metric].max()
        axes[i].set_ylim(y_min * (1 - Y_PAD), y_max * (1 + Y_PAD))
        axes[i].set_ylabel(metric)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


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
