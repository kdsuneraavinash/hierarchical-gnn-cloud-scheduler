import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_summary_radar_chart(df: pd.DataFrame) -> None:
    _, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Compute the average values per scheduler
    avg_df = df.groupby("name", as_index=False).agg(
        {"makespan": "mean", "energy_consumption": "mean", "sla_penalty": "mean"}
    )

    # Normalize the data for better comparison
    def normalize(data):
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        return (data - min_vals) / (max_vals - min_vals + 1e-8)

    metrics = ["makespan", "energy_consumption", "sla_penalty"]
    normalized_values = normalize(avg_df[metrics].values)

    # Compute angles for the radar chart
    num_vars = len(metrics)
    angles: list[float] = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()  # type: ignore
    angles += angles[:1]  # Close the radar chart loop

    colors = plt.cm.get_cmap("tab10", len(avg_df["name"])).colors  # type: ignore

    # Plot each scheduler's data
    for idx, (scheduler, values) in enumerate(zip(avg_df["name"], normalized_values)):
        values = np.append(values, values[0])  # Close the loop
        ax.plot(angles, values, label=scheduler, linewidth=2, marker="o", color=colors[idx])
        ax.fill(angles, values, alpha=0.2, color=colors[idx])

    # Set labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_title("Scheduler Performance Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
