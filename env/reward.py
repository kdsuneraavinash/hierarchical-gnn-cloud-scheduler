from env.simulation import Simulation


class RewardFunction:
    """Handles reward calculation"""

    prev_makespan: float
    prev_energy_consumption: float
    prev_sla_penalty: float

    def __init__(self, makespan_alpha: float = 1, energy_alpha: float = 1, sla_penalty_alpha: float = 1):
        self.makespan_alpha = makespan_alpha
        self.energy_alpha = energy_alpha
        self.sla_penalty_alpha = sla_penalty_alpha

    def next_episode(self, simulation: Simulation) -> None:
        self.prev_makespan = simulation.makespan()
        self.prev_energy_consumption = simulation.total_energy_consumption()
        self.prev_sla_penalty = simulation.total_sla_penalty()

    def current_reward(self, simulation: Simulation, done: bool) -> float:
        """Computes the reward based on makespan and energy consumption."""
        curr_makespan = simulation.makespan()
        curr_energy_consumption = simulation.total_energy_consumption()
        curr_sla_penalty = simulation.total_sla_penalty()

        makespan_reward_diff = (curr_makespan - self.prev_makespan) / curr_makespan
        energy_consumption_reward_diff = (
            curr_energy_consumption - self.prev_energy_consumption
        ) / curr_energy_consumption
        sla_penalty_reward_diff = (curr_sla_penalty - self.prev_sla_penalty) / curr_sla_penalty

        preference = simulation.dataset.preference
        total_preference = preference.makespan + preference.energy_consumption + preference.sla_penalty
        reward = -(
            self.makespan_alpha * makespan_reward_diff * preference.makespan / total_preference
            + self.energy_alpha * energy_consumption_reward_diff * preference.energy_consumption / total_preference
            + self.sla_penalty_alpha * sla_penalty_reward_diff * preference.sla_penalty / total_preference
        )

        self.prev_makespan = curr_makespan
        self.prev_energy_consumption = curr_energy_consumption
        self.prev_sla_penalty = curr_sla_penalty
        return reward
