from env.simulation import Simulation


class RewardFunction:
    """Handles reward calculation"""

    prev_makespan: float
    prev_energy_consumption: float
    prev_latency_score: float

    def __init__(self, makespan_alpha: float = 1, energy_alpha: float = 1, latency_alpha: float = 1):
        self.makespan_alpha = makespan_alpha
        self.energy_alpha = energy_alpha
        self.latency_alpha = latency_alpha

    def next_episode(self, simulation: Simulation) -> None:
        self.prev_makespan = simulation.makespan()
        self.prev_energy_consumption = simulation.total_energy_consumption()
        self.prev_latency_score = simulation.total_latency_score()

    def current_reward(self, simulation: Simulation, done: bool) -> float:
        """Computes the reward based on makespan and energy consumption."""
        curr_makespan = simulation.makespan()
        curr_energy_consumption = simulation.total_energy_consumption()
        curr_latency_score = simulation.total_latency_score()

        makespan_reward_diff = (curr_makespan - self.prev_makespan) / curr_makespan
        energy_consumption_reward_diff = (
            curr_energy_consumption - self.prev_energy_consumption
        ) / curr_energy_consumption
        latency_score_reward_diff = (curr_latency_score - self.prev_latency_score) / curr_latency_score

        preference = simulation.dataset.preference
        total_preference = preference.makespan + preference.energy_consumption + preference.latency_score
        reward = -(
            self.makespan_alpha * makespan_reward_diff * preference.makespan / total_preference
            + self.energy_alpha * energy_consumption_reward_diff * preference.energy_consumption / total_preference
            + self.latency_alpha * latency_score_reward_diff * preference.latency_score / total_preference
        )

        self.prev_makespan = curr_makespan
        self.prev_energy_consumption = curr_energy_consumption
        self.prev_latency_score = curr_latency_score
        return reward
