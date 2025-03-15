from env.simulation import Simulation


class RewardFunction:
    """Handles reward calculation"""

    prev_makespan: float
    prev_energy_consumption: float

    def __init__(self, makespan_alpha: float = 0.3, energy_alpha: float = 4):
        self.makespan_alpha = makespan_alpha
        self.energy_alpha = energy_alpha

    def next_episode(self, simulation: Simulation) -> None:
        self.prev_makespan = simulation.makespan()
        self.prev_energy_consumption = simulation.total_energy_consumption()

    def current_reward(self, simulation: Simulation, done: bool) -> float:
        """Computes the reward based on makespan and energy consumption."""
        curr_makespan = simulation.makespan()
        curr_energy_consumption = simulation.total_energy_consumption()

        makespan_reward_diff = (curr_makespan - self.prev_makespan) / curr_makespan
        energy_consumption_reward_diff = (
            curr_energy_consumption - self.prev_energy_consumption
        ) / curr_energy_consumption

        preference = simulation.dataset.preference
        reward = -(
            self.makespan_alpha * makespan_reward_diff * preference.makespan
            + self.energy_alpha * energy_consumption_reward_diff * preference.energy_consumption
        )

        self.prev_makespan = curr_makespan
        self.prev_energy_consumption = curr_energy_consumption
        return reward
