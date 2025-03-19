from typing import Any
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
import seaborn as sns

from constants import CHART_AXIS_PAD


def plot_mo_summary(summary_data: dict[str, list[dict[str, float]]]) -> None:
    pareto_point_map: dict[str, np.ndarray[tuple[int, ...], Any]] = {}
    run_time_map: dict[str, float] = {}
    for scheduler, results in summary_data.items():
        latency_score_results = [result["latency_score"] for result in results]
        makespan_results = np.array([result["makespan"] for result in results])
        energy_consumption_results = np.array([result["energy_consumption"] for result in results])
        latency_score_results = np.array([result["latency_score"] for result in results])
        run_time_results = np.array([result["run_time"] for result in results])

        points = np.column_stack((makespan_results, energy_consumption_results, latency_score_results))
        pareto_indices = find_pareto_front(points)
        run_time_map[scheduler] = run_time_results.sum()
        pareto_point_map[scheduler] = np.array(
            [
                (
                    makespan_results[pareto_index],
                    energy_consumption_results[pareto_index],
                    latency_score_results[pareto_index],
                )
                for pareto_index in pareto_indices
            ]
        )

    all_points = np.vstack([results for results in pareto_point_map.values()])
    worst_values = np.max(all_points, axis=0)
    reference_point = worst_values * 1.1
    ref_pareto_indices = find_pareto_front(all_points)
    ref_pareto_points = all_points[ref_pareto_indices]
    hv = HV(ref_point=reference_point)
    gd = GD(ref_pareto_points)
    igd = IGD(ref_pareto_points)

    data: list[dict[str, Any]] = []
    for scheduler, pareto_points in pareto_point_map.items():
        data.append(
            {
                "name": scheduler,
                "hypervolume": hv(pareto_points),
                "gd": gd(pareto_points),
                "igd": igd(pareto_points),
                "run_time": run_time_map[scheduler],
            }
        )

    df = pd.DataFrame(data)
    print(df)

    df["proposed"] = df["name"] == "Proposed"

    fig, axes_ = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    axes: list[Axes] = list(axes_.flatten())

    for i, metric in enumerate(["hypervolume", "gd", "igd", "run_time"]):
        avg_sorted = df.sort_values(metric)
        sns.barplot(data=avg_sorted, x="name", y=metric, hue="proposed", ax=axes[i], palette="Set2", legend=False)
        y_min, y_max = avg_sorted[metric].min(), avg_sorted[metric].max()
        axes[i].set_ylim(y_min * (1 - CHART_AXIS_PAD), y_max * (1 + CHART_AXIS_PAD))
        axes[i].set_ylabel(metric)
        axes[i].xaxis.set_ticks(avg_sorted["name"].unique())
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
