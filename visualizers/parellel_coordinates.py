from matplotlib import axes
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler


def plot_parallel_coordinates(ax: axes.Axes, df: pd.DataFrame) -> None:
    avg_df = df.groupby("name", as_index=False).agg(
        {"makespan": "mean", "energy_consumption": "mean", "sla_penalty": "mean"}
    )
    # Maintain the original order of appearance in df
    avg_df["name"] = pd.Categorical(avg_df["name"], categories=df["name"].dropna().unique(), ordered=True)
    avg_df = avg_df.sort_values("name").reset_index(drop=True)

    scaler = MinMaxScaler()
    avg_df[["makespan", "energy_consumption", "sla_penalty"]] = scaler.fit_transform(
        avg_df[["makespan", "energy_consumption", "sla_penalty"]]
    )
    parallel_coordinates(avg_df, class_column="name", colormap="tab10", alpha=0.7, ax=ax)
