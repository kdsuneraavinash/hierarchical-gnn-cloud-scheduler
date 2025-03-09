import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


def plot_summary_chart(ax: plt.Axes, df: pd.DataFrame) -> None:
    # Compute the average makespan and energy consumption per name
    avg_df = df.groupby("name", as_index=False).agg({"makespan": "mean", "energy_consumption": "mean"})

    # Maintain the original order of appearance in df
    avg_df["name"] = pd.Categorical(avg_df["name"], categories=df["name"].dropna().unique(), ordered=True)
    avg_df = avg_df.sort_values("name").reset_index(drop=True)

    # Create a secondary Y-axis
    ax_ = ax.twinx()

    # Bar plot for Makespan on the primary Y-axis
    ax.bar(avg_df["name"], avg_df["makespan"], color="#dffdb9", label="Makespan", edgecolor="black")
    ax.set_ylabel("Makespan (s)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    # Line plot for Energy Consumption on the secondary Y-axis
    ax_.plot(avg_df["name"], avg_df["energy_consumption"], color="#ff5757", marker="o", label="Energy Consumption")
    ax_.set_ylabel("Energy Consumption (J)")
    ax_.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    # Add legends
    ax.legend(loc="upper left")
    ax_.legend(loc="upper right")

    # X-axis formatting
    ax.set_xlabel("Name")
    ax.set_xticks(range(len(avg_df["name"])))
    ax.set_xticklabels(avg_df["name"], rotation=45, ha="right")
