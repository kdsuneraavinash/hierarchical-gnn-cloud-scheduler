import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from constants import CHART_AXIS_PAD


def plot_summary_charts(df: pd.DataFrame) -> None:
    avg_df = df.groupby("name", as_index=False).agg(
        {"makespan": "mean", "energy_consumption": "mean", "sla_penalty": "mean"}
    )

    avg_df["proposed"] = avg_df["name"].str.startswith("Proposed")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    for i, metric in enumerate(["makespan", "energy_consumption", "sla_penalty"]):
        avg_sorted = avg_df.sort_values(metric)
        sns.barplot(data=avg_sorted, x="name", y=metric, hue="proposed", ax=axes[i], palette="Set2", legend=False)
        y_min, y_max = avg_sorted[metric].min(), avg_sorted[metric].max()
        axes[i].set_ylim(y_min * (1 - CHART_AXIS_PAD), y_max * (1 + CHART_AXIS_PAD))
        axes[i].set_ylabel(metric)
        axes[i].xaxis.set_ticks(avg_sorted["name"].unique())
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()
