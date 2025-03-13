import numpy as np
from env.simulation import Simulation


class RewardFunction:
    """Handles reward calculation"""

    makespan_diffs: list[float]
    energy_consumption_diffs: list[float]
    sla_penalty_diffs: list[float]
    final_makespans: list[float] = []
    final_energy_consumptions: list[float] = []
    final_sla_penalties: list[float] = []

    prev_makespan: float
    prev_energy_consumption: float
    prev_sla_penalty: float

    def next_episode(self, simulation: Simulation) -> None:
        self.makespan_diffs = []
        self.energy_consumption_diffs = []
        self.sla_penalty_diffs = []
        self.prev_makespan = simulation.makespan()
        self.prev_energy_consumption = simulation.total_energy_consumption()
        self.prev_sla_penalty = simulation.total_sla_penalty()

    def current_reward(self, simulation: Simulation, done: bool) -> float:
        """Computes the reward based on makespan, energy consumption, and SLA penalties."""
        curr_makespan = simulation.makespan()
        curr_energy_consumption = simulation.total_energy_consumption()
        curr_sla_penalty = simulation.total_sla_penalty()

        makespan_reward_diff = curr_makespan - self.prev_makespan
        energy_consumption_reward_diff = curr_energy_consumption - self.prev_energy_consumption
        sla_penalty_reward_diff = curr_sla_penalty - self.prev_sla_penalty

        self.makespan_diffs.append(makespan_reward_diff)
        self.energy_consumption_diffs.append(energy_consumption_reward_diff)
        self.sla_penalty_diffs.append(sla_penalty_reward_diff)

        preference = simulation.dataset.preference
        reward = -(
            makespan_reward_diff * preference.makespan
            + energy_consumption_reward_diff * preference.energy_consumption
            + sla_penalty_reward_diff * preference.sla_penalty
        )

        if done:
            self.final_makespans.append(curr_makespan)
            self.final_energy_consumptions.append(curr_energy_consumption)
            self.final_sla_penalties.append(curr_sla_penalty)

            makespan_lambda = np.mean(self.makespan_diffs) / np.mean(self.final_makespans)
            energy_consumption_lambda = np.mean(self.energy_consumption_diffs) / np.mean(self.final_energy_consumptions)
            sla_penalty_lambda = np.mean(self.sla_penalty_diffs) / np.mean(self.final_sla_penalties)
            reward -= float(
                makespan_lambda * curr_makespan * preference.makespan
                + energy_consumption_lambda * curr_energy_consumption * preference.energy_consumption
                + sla_penalty_lambda * curr_sla_penalty * preference.sla_penalty
            )

        self.prev_makespan = curr_makespan
        self.prev_energy_consumption = curr_energy_consumption
        self.prev_sla_penalty = curr_sla_penalty
        return reward
