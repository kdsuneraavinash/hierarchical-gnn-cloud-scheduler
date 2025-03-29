from typing import Any
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
import seaborn as sns

from constants import CHART_AXIS_PAD, SCH_COLORS


dynamic_schedulers = {
    "Random",
    "Least Loaded First",
    "Round--Robin",
    "Weighted Dynamic",
    "Proposed",
    "Proposed-Syn",
    "Proposed-Real",
}


def plot_mo_summary(summary_data: dict[str, list[dict[str, float]]], num_tasks: int, dataset_name: str) -> None:
    pareto_point_map: dict[str, np.ndarray[tuple[int, ...], Any]] = {}
    direct_metrics: dict[str, dict[str, float]] = {}
    for scheduler, results in summary_data.items():
        sla_penalty_results = [result["sla_penalty"] for result in results]
        makespan_results = np.array([result["makespan"] for result in results])
        energy_consumption_results = np.array([result["energy_consumption"] for result in results])
        sla_penalty_results = np.array([result["sla_penalty"] for result in results])
        run_time_results = np.array([result["run_time"] for result in results])

        points = np.column_stack((makespan_results, energy_consumption_results, sla_penalty_results))
        pareto_indices = find_pareto_front(points)
        significant_run_times = np.array([t for t in run_time_results if t > 1e-4])
        direct_metrics[scheduler] = {
            "run_time": significant_run_times.sum(),
            "decision_latency": significant_run_times.mean() / (num_tasks if scheduler in dynamic_schedulers else 1),
        }
        pareto_point_map[scheduler] = np.array(
            [
                (
                    makespan_results[pareto_index],
                    energy_consumption_results[pareto_index],
                    sla_penalty_results[pareto_index],
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
    gd_plus = GDPlus(ref_pareto_points)
    igd_plus = IGDPlus(ref_pareto_points)

    data: list[dict[str, Any]] = []
    for scheduler, pareto_points in pareto_point_map.items():
        data.append(
            {
                "name": scheduler,
                "hypervolume": hv(pareto_points),
                "gd": gd(pareto_points),
                "igd": igd(pareto_points),
                "gd_plus": gd_plus(pareto_points),
                "igd_plus": igd_plus(pareto_points),
                "run_time": direct_metrics[scheduler]["run_time"],
                "decision_latency": direct_metrics[scheduler]["decision_latency"],
            }
        )

    df = pd.DataFrame(data)
    print(df)

    df["proposed"] = df["name"].str.startswith("Proposed")

    fig, axes_ = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    axes: list[Axes] = list(axes_.flatten())

    metrics = [
        # metric key, log scale
        ("hypervolume", "Hypervolume", False),
        # ("gd", "GD", False),
        ("igd", "IGD", False),
        # ("gd_plus", "GD+", False),
        # ("igd_plus", "IGD+", False),
        # ("run_time", "Run time", False),
        ("decision_latency", "Decision Latency (Log Scale)", True),
    ]
    for i, (metric, y_axis, log_scale) in enumerate(metrics):
        # avg_sorted = df.sort_values(metric)
        sns.barplot(data=df, x="name", y=metric, hue="name", ax=axes[i], palette=SCH_COLORS, legend=False)
        y_min, y_max = df[metric].min(), df[metric].max()
        if log_scale:
            axes[i].set_yscale("log")
        else:
            axes[i].set_ylim(y_min * (1 - CHART_AXIS_PAD), y_max * (1 + CHART_AXIS_PAD))
        axes[i].set_ylabel(y_axis)
        axes[i].set_xlabel("Scheduler")
        axes[i].set_title(f"{dataset_name} - {y_axis}")
        axes[i].xaxis.set_ticks(df["name"].unique())
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")

    # Uncomment to output latex table part
    # print()
    # for _, row in df.iterrows():
    #     values = list(map(lambda v: f"{v:.3f}", [row["hypervolume"], row["igd"], row["decision_latency"]]))
    #     values.insert(0, row["name"])
    #     print(" & ".join(values))
    # print()

    return fig


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
