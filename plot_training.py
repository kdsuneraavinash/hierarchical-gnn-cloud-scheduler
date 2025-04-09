from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from visualizers.training_plot import training_plot


log_dirs: list[tuple[int, int, int, str]] = [
    # (1, 1, 1, "logs/1743004795_gnn_synthetic_[1][1][1]"),
    (0, 1, 1, "logs/1743008150_gnn_synthetic_[0][1][1]"),
    (1, 0, 1, "logs/1743011666_gnn_synthetic_[1][0][1]"),
    (1, 1, 0, "logs/1743014991_gnn_synthetic_[1][1][0]"),
    (1, 0, 0, "logs/1743025325_gnn_synthetic_[1][0][0]"),
    (0, 1, 0, "logs/1743021872_gnn_synthetic_[0][1][0]"),
    (0, 0, 1, "logs/1743018436_gnn_synthetic_[0][0][1]"),
]


def get_df(path: str):
    path_ = Path(__file__).parent / path
    tfevents_file: Path | None = None
    for file in path_.iterdir():
        if file.name.startswith("events.out.tfevents"):
            tfevents_file = file
            break
    if tfevents_file is None:
        raise ValueError(f"No log file found inside {path_}")

    ea = event_accumulator.EventAccumulator(
        str(tfevents_file),
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 200000,
            event_accumulator.HISTOGRAMS: 0,
        },
    )
    ea.Reload()

    return pd.DataFrame(ea.Scalars("charts/episodic_return"))


def main():
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    axes = axes.flatten()
    for (m, e, s, log_dir), ax in zip(log_dirs, axes):
        df = get_df(log_dir)
        training_plot(df, ax)
        ax.set_title(f"$\\lambda_1 = {m}, \\lambda_2 = {e}, \\lambda_3 = {s}$")
    plt.show()


if __name__ == "__main__":
    main()
