from env.simulation import Simulation


class RewardFunction:
    """Handles reward calculation"""

    prev_makespan: float
    prev_energy_consumption: float
    prev_sla_penalty: float

    def next_episode(self, simulation: Simulation) -> None:
        self.prev_makespan = simulation.makespan()
        self.prev_energy_consumption = simulation.total_energy_consumption()
        self.prev_sla_penalty = simulation.total_sla_penalty()

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

        preference = simulation.dataset.preference
        reward = -(
            makespan_reward_diff * preference.makespan
            + energy_consumption_reward_diff * preference.energy_consumption
            + sla_penalty_reward_diff * preference.sla_penalty
        )

        self.prev_makespan = curr_makespan
        self.prev_energy_consumption = curr_energy_consumption
        self.prev_sla_penalty = curr_sla_penalty
        return reward
