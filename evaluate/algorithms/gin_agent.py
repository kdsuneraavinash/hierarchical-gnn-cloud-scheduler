import torch

from dataset.models import Dataset
from env.observation import create_env_obs, encode_env_obs
from env.simulation import SimulationState
from evaluate.algorithms.base_dynamic import BaseDynamicScheduler
from models.agent import GinAgent


class GinAgentScheduler(BaseDynamicScheduler):
    def __init__(self, name: str, model_path: str | None = None, agent: GinAgent | None = None):
        super().__init__(name)

        if agent is not None:
            self.agent = agent
        elif model_path is not None:
            self.agent = GinAgent(device=torch.device("cpu"))
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
        encoded_obs_tensor = torch.Tensor(encoded_obs, device=self.agent.device)
        action = self.agent.get_action_unbatched(encoded_obs_tensor)
        task_id = int(action.item()) // len(dataset.vms)
        vm_id = int(action.item()) % len(dataset.vms)

        return task_id, vm_id
