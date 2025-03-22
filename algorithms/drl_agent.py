import torch

from algorithms.base_dynamic import BaseDynamicScheduler
from constants import N_VM
from dataset.models import Dataset
from env.observation import create_env_obs, encode_env_obs
from env.state import SimulationState
from models.agent import make_agent
from models.base_agent import BaseAgent


class DrlAgentScheduler(BaseDynamicScheduler):
    def __init__(
        self,
        name: str,
        model_path: str | None = None,
        agent_type: str | None = None,
        agent: BaseAgent | None = None,
    ):
        super().__init__(name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if agent is not None:
            self.agent = agent
        elif model_path is not None:
            self.agent = make_agent(agent_type, device)
            self.agent.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            raise ValueError("Must provide one of model path or agent")

    def select_task_and_vm(self, dataset: Dataset, state: SimulationState) -> tuple[int, int]:
        obs = create_env_obs(
            dataset=dataset,
            task_states=state.task_states,
            vm_states=state.vm_states,
            task_dependencies=state.task_dependencies,
        )
        encoded_obs = encode_env_obs(obs)
        encoded_obs_tensor = torch.Tensor(encoded_obs).to(self.agent.device)
        action, _, _, _ = self.agent.get_action_and_value(encoded_obs_tensor.unsqueeze(0))
        task_id = int(action.item()) // N_VM
        vm_id = int(action.item()) % N_VM

        return task_id, vm_id
