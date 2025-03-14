import numpy as np
from env.simulation import Simulation


class RewardFunction:
    """Handles reward calculation"""

    prev_makespan: float
    prev_energy_consumption: float
    prev_sla_penalty: float
    diff_histories: dict[str, list[float]]

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size

    def next_episode(self, simulation: Simulation) -> None:
        self.prev_makespan = simulation.makespan()
        self.prev_energy_consumption = simulation.total_energy_consumption()
        self.prev_sla_penalty = simulation.total_sla_penalty()
        self.diff_histories = {}

    def current_reward(self, simulation: Simulation, done: bool) -> float:
        """Computes the reward based on makespan, energy consumption, and SLA penalties."""
        curr_makespan = simulation.makespan()
        curr_energy_consumption = simulation.total_energy_consumption()
        curr_sla_penalty = simulation.total_sla_penalty()

        makespan_reward_diff = (curr_makespan - self.prev_makespan) / curr_makespan
        energy_consumption_reward_diff = (
            curr_energy_consumption - self.prev_energy_consumption
        ) / curr_energy_consumption
        sla_penalty_reward_diff = (curr_sla_penalty - self.prev_sla_penalty) / curr_sla_penalty

        makespan_reward_norm = self.normalize("makespan", makespan_reward_diff)
        energy_consumption_reward_norm = self.normalize("energy_consumption", energy_consumption_reward_diff)
        sla_penalty_reward_norm = self.normalize("sla_penalty", sla_penalty_reward_diff)

        preference = simulation.dataset.preference
        reward = -(
            makespan_reward_norm * preference.makespan
            + energy_consumption_reward_norm * preference.energy_consumption
            + sla_penalty_reward_norm * preference.sla_penalty
        )

        self.prev_makespan = curr_makespan
        self.prev_energy_consumption = curr_energy_consumption
        self.prev_sla_penalty = curr_sla_penalty
        return reward

    def normalize(self, param_name: str, diff: float) -> float:
        if param_name not in self.diff_histories:
            self.diff_histories[param_name] = []

        self.diff_histories[param_name].append(diff)
        if len(self.diff_histories[param_name]) > self.history_size:
            self.diff_histories[param_name].pop(0)

        norm = diff / (np.mean(self.diff_histories[param_name]) + 1e-8)
        return float(norm)
