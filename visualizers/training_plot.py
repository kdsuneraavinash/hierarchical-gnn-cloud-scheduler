from matplotlib import pyplot as plt
import pandas as pd


def training_plot(import_csv: str, max_steps: int = 100_000, episode_length: int = 50, num_envs: int = 4):
    data = pd.read_csv(import_csv)
    _steps = data["Step"]

    capped_data = data[_steps <= max_steps]
    capped_steps = capped_data["Step"] / (episode_length * num_envs)
    capped_episodic_returns = capped_data["Value"]
    ema_episodic_returns = capped_episodic_returns.ewm(span=100).mean()

    plt.figure(figsize=(10, 6))
    # plt.plot(capped_steps, capped_episodic_returns, color="lightcoral", label="Original Episodic Return", alpha=0.5)
    plt.plot(capped_steps, ema_episodic_returns, color="red", label="Smoothed Episodic Return", linewidth=1)

    plt.title("Training Curve")
    plt.xlabel("Number of Steps")
    plt.ylabel("Episodic Return")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    training_plot("/home/kdsuneraavinash/Downloads/1742091521_gnn_[1][1][1].csv")
