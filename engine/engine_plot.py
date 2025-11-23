# Replot with distinct colors for RL (Proposed) and Default (Current)
import matplotlib.pyplot as plt
import numpy as np

# Data remains the same
labels = ["Proposed (RL)", "Current (Default)"]

data = {
    "Execution Time": [[101, 84, 61, 159, 68], [120, 128, 74, 222, 117]],
    "Energy Consumption": [
        [208.31569113756615, 338.74428104575156, 572.4756382117287, 337.44505718954235, 277.65844896933123],
        [363.15229719128433, 338.74428104575156, 572.4756382117288, 337.44505718954235, 281.67178900620064],
    ],
    "SLA Penalty": [[949, 1010, 431, 1872, 1061], [1501, 1650, 781, 2897, 1320]],
}

means = {k: [np.mean(v[0]), np.mean(v[1])] for k, v in data.items()}
stds = {k: [np.std(v[0]), np.std(v[1])] for k, v in data.items()}

# Colors for RL and Default
colors = ["#2E8B57", "#B22222"]  # green for RL, red for Default

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Comparison of Proposed vs Current Scheduler", fontsize=14, fontweight="bold")

for ax, metric in zip(axes, data.keys()):
    x = np.arange(len(labels))
    bars = ax.bar(x, means[metric], yerr=stds[metric], capsize=8, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02 * height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
