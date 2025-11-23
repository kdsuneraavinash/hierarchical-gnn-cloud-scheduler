from matplotlib import axes
import pandas as pd

from constants import CHART_AXIS_PAD


def training_plot(df: pd.DataFrame, ax: axes.Axes, episode_length: int = 50, num_envs: int = 4):
    capped_steps = df["step"] / (episode_length * num_envs)
    capped_episodic_returns = df["value"]
    ema_episodic_returns = capped_episodic_returns.ewm(span=10).mean()

    ax.plot(capped_steps, capped_episodic_returns, color="lightcoral", label="Original Episodic Return", alpha=0.2)
    ax.plot(capped_steps, ema_episodic_returns, color="red", label="Smoothed Episodic Return", linewidth=1)

    ax.set_ylim(ema_episodic_returns.min() - CHART_AXIS_PAD, ema_episodic_returns.max() + CHART_AXIS_PAD)
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Episodic Return")
    ax.grid(True)
