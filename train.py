# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.wrappers import RecordEpisodeStatistics
from progress_table import ProgressTable
from progress_table.v1.progress_table import TableProgressBar
from torch.utils.tensorboard.writer import SummaryWriter

from algorithms.drl_agent import DrlAgentScheduler
from constants import (
    ENERGY_CONSUMPTION_PREFERENCE,
    LATENCY_SCORE_PREFERENCE,
    MAKESPAN_PREFERENCE,
    N_HOST,
    N_TASK,
    N_VM,
    TEST_SEED,
    WORKFLOW_TASKS,
)
from dataset.generator import generate_dataset
from dataset.models import Solution
from dataset.real_world import RealWorldDatasetArgs
from env.gym_env import GymEnvironment
from models.agent import make_agent
from models.base_agent import BaseAgent


@dataclass
class Args:
    exp_name: str = "test"
    """the name of this experiment"""

    agent_type: str = "gnn"
    """the type of agent (gnn, drl)"""

    seed: int = 1
    """seed of the experiment"""
    output_dir: str = "logs"
    """the output directory of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str | None = None
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    load_model_dir: str | None = None
    """Directory to load the model from"""
    test_iterations: int = 1
    """number of test iterations"""

    # Algorithm specific arguments
    total_timesteps: int = 200_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    dataset: RealWorldDatasetArgs = field(
        default_factory=lambda: RealWorldDatasetArgs(
            task_count=N_TASK,
            max_vm_count=N_VM,
            max_host_count=N_HOST,
            tasks_per_workflow=WORKFLOW_TASKS,
            makespan_preference=MAKESPAN_PREFERENCE,
            energy_consumption_preference=ENERGY_CONSUMPTION_PREFERENCE,
            latency_score_preference=LATENCY_SCORE_PREFERENCE,
        )
    )
    """the dataset generation parameters"""
    test_dataset: RealWorldDatasetArgs = field(
        default_factory=lambda: RealWorldDatasetArgs(
            task_count=N_TASK,
            max_vm_count=N_VM,
            max_host_count=N_HOST,
            tasks_per_workflow=WORKFLOW_TASKS,
            makespan_preference=MAKESPAN_PREFERENCE,
            energy_consumption_preference=ENERGY_CONSUMPTION_PREFERENCE,
            latency_score_preference=LATENCY_SCORE_PREFERENCE,
        )
    )
    """the test dataset generation parameters"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    run_name: str = ""
    """the full name of the run"""


# Environment creation
# ----------------------------------------------------------------------------------------------------------------------


def make_env(idx: int, args: Args) -> gym.Env[np.ndarray[tuple[int, ...], Any], np.int64]:
    env = GymEnvironment(dataset_args=args.dataset)
    return RecordEpisodeStatistics(env)


# Training Agent
# ----------------------------------------------------------------------------------------------------------------------


def train(args: Args) -> None:
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.run_name = f"{int(time.time())}_{args.exp_name}"
    if args.track:
        import wandb

        assert args.wandb_project_name is not None, "Please specify the wandb project name"
        assert args.wandb_entity is not None, "Please specify the entity of wandb project"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"{args.output_dir}/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using {device} for training...")

    # env setup
    envs = gym.vector.SyncVectorEnv([lambda: make_env(i, args) for i in range(args.num_envs)])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert obs_space.shape is not None
    assert act_space.shape is not None

    agent = make_agent(args.agent_type, device)
    writer.add_text("agent", f"```{agent}```")

    last_model_save = 0
    if args.load_model_dir:
        model_path = Path(__file__).parent.parent.parent / "logs" / args.load_model_dir / "model.pt"
        agent.load_state_dict(torch.load(str(model_path), weights_only=True))
        print(f"Loaded model from {model_path}")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Variables used to compute returns
    next_obs: np.ndarray[tuple[int, ...], Any]
    reward: np.ndarray[tuple[int, ...], Any]
    terminations: np.ndarray[tuple[int, ...], Any]
    truncations: np.ndarray[tuple[int, ...], Any]
    infos: dict[str, Any]
    approx_kl = v_loss = pg_loss = entropy_loss = old_approx_kl = approx_kl = None

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_tensor = torch.Tensor(next_obs).to(device)
    next_done_tensor = torch.zeros(args.num_envs).to(device)

    table = ProgressTable(print_header_every_n_rows=0, pbar_embedded=False, pbar_show_eta=True)
    progress_bar: TableProgressBar = table.pbar(range(args.total_timesteps))
    table.add_column("iter")
    table.add_column("global_step")
    table.add_column("phase", width=10)
    table.add_column("reward", width=10)
    table.add_column("t_makespan", width=10)
    table.add_column("t_energy_consumption", width=10)
    table.add_column("t_latency_score", width=10)

    for iteration in range(1, args.num_iterations + 1):
        table.update("iter", value=iteration)
        table.update("global_step", value=global_step)
        table.update("phase", value="running")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs_tensor
            dones[step] = next_done_tensor

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.Tensor(reward).to(device).view(-1)
            next_obs_tensor, next_done_tensor = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "episode" in infos:
                for i in range(args.num_envs):
                    writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)
                    writer.add_scalar("episode/makespan", infos["makespan"][i], global_step)
                    writer.add_scalar("episode/energy_consumption", infos["energy_consumption"][i], global_step)
                    writer.add_scalar("episode/latency_score", infos["latency_score"][i], global_step)
                    table.update("reward", value=infos["episode"]["r"][i], aggregate="mean")

        progress_bar.set_step(global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_tensor).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done_tensor
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            assert approx_kl is not None
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        assert v_loss is not None and pg_loss is not None and entropy_loss is not None
        assert old_approx_kl is not None and approx_kl is not None
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        with torch.no_grad():
            table.update("phase", value="testing")
            test_results = test_agent(agent, args)
            writer.add_scalar("tests/makespan", test_results[0], global_step)
            writer.add_scalar("tests/energy_consumption", test_results[1], global_step)
            writer.add_scalar("tests/latency_score", test_results[2], global_step)
            table.update("t_makespan", value=test_results[0])
            table.update("t_energy_consumption", value=test_results[1])
            table.update("t_latency_score", value=test_results[2])

        table.update("phase", value="done")
        if (global_step - last_model_save) >= 10_000:
            table.update("phase", value="saved")
            torch.save(agent.state_dict(), f"{args.output_dir}/{args.run_name}/model_{global_step}.pt")
            last_model_save = global_step

        table.next_row()

    torch.save(agent.state_dict(), f"{args.output_dir}/{args.run_name}/model.pt")

    envs.close()
    writer.close()

    table.close()


# Testing Agent
# ----------------------------------------------------------------------------------------------------------------------


def test_agent(agent: BaseAgent, args: Args) -> tuple[float, float, float]:
    test_rng = np.random.RandomState(TEST_SEED)

    total_makespan = 0.0
    total_energy_consumption = 0.0
    total_latency_score = 0.0

    for _ in range(args.test_iterations):
        dataset = generate_dataset(args.test_dataset, test_rng)

        test_scheduler = DrlAgentScheduler(name="Agent", agent=agent, agent_type=args.agent_type)
        assignments = test_scheduler.schedule(dataset)
        solution = Solution(dataset, assignments)

        total_makespan += solution.makespan()
        total_energy_consumption += solution.energy_consumption()
        total_latency_score += solution.latency_score()

    avg_makespan = total_makespan / args.test_iterations
    avg_energy_consumption = total_energy_consumption / args.test_iterations
    avg_latency_score = total_latency_score / args.test_iterations
    return avg_makespan, avg_energy_consumption, avg_latency_score


if __name__ == "__main__":
    train(tyro.cli(Args))
