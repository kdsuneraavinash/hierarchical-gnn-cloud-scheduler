import torch

from dataset.models import Dataset, VmAssignment
from env.observation import create_env_obs, encode_env_obs
from env.simulation import Simulation
from evaluate.algorithms.base import BaseScheduler
from models.agent import GinAgent


class AgentScheduler(BaseScheduler):
    def __init__(self, model_path: str | None = None, agent: GinAgent | None = None):
        super().__init__("Proposed")

        if agent is not None:
            self.agent = agent
        elif model_path is not None:
            self.agent = GinAgent(device=torch.device("cpu"))
            self.agent.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            raise ValueError("Must provide one of model path or agent")

    def schedule(self, dataset: Dataset) -> list[VmAssignment]:
        simulation = Simulation(dataset)

        done = False
        while not done:
            obs = create_env_obs(
                dataset=simulation.dataset,
                task_states=simulation.task_states,
                vm_states=simulation.vm_states,
                task_dependencies=simulation.task_dependencies,
            )
            encoded_obs = encode_env_obs(obs)
            encoded_obs_tensor = torch.Tensor(encoded_obs, device=self.agent.device)
            action = self.agent.get_action_unbatched(encoded_obs_tensor)

            task_id = int(action.item()) // len(dataset.vms)
            vm_id = int(action.item()) % len(dataset.vms)
            error, done = simulation.assign_vm(task_id, vm_id)
            if error:
                raise ValueError(error)

        return simulation.to_assignments()
