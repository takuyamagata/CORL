import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import d4rl.gym_mujoco
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import scipy

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cpu" #"cuda"
    env: str = "hopper-medium-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 5_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount_base: float = 0.99  # Base discount factor
    discount: float = 0.999  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 10.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.9  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = False  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    critic_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    exp_order: int = 4 # Expansion order
    lr_warm_up_period: int = 0 # Learning rate annealing initial warmup period
    training_stage_periods = 100000 # number of iterations for each seq. component training 
    # Wandb logging
    project: str = "CORL"
    group: str = "SeqIQL-D4RL"
    name: str = "SeqIQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


# @torch.no_grad()
# def eval_actor(
#     env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int,
# ) -> np.ndarray:
#     env.seed(seed)
#     actor.eval()
#     episode_rewards = []
#     for _ in range(n_episodes):
#         state, done = env.reset(), False
#         episode_reward = 0.0
#         while not done:
#             action = actor.act(state, device)
#             state_, reward, done, _ = env.step(action)
#             episode_reward += reward
#             state = state_
#         episode_rewards.append(episode_reward)

#     actor.train()
#     return np.asarray(episode_rewards)

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int, rec_trajectory: bool = False, rec_annomaly_score: bool = False, dataset: Dict[str, np.ndarray] = {}
) -> Tuple[np.ndarray, List[TensorBatch]]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    trajectory = []
    anomaly_score = []
    if rec_annomaly_score:
        sa_dataset = np.hstack((dataset['observations'], dataset['actions']))
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        # s, a, r, s_, d = [], [], [], [], []
        cnt = 0
        while not done:
            action = actor.act(state, device)
            state_, reward, done, _ = env.step(action)
            episode_reward += reward
            # if rec_trajectory:
            #     s.append(state); a.append(action); r.append(reward); s_.append(state_); d.append(done)
            if rec_annomaly_score and (cnt % 5 == 0):
                anomaly_score.append(compute_annomaly_score(sa_dataset, np.hstack((state, action))))
            state = state_
            cnt = cnt + 1
        episode_rewards.append(episode_reward)
        # trajectory.append([s, a, r, s_, d])

    actor.train()
    return np.asarray(episode_rewards), trajectory, np.asarray(anomaly_score)

def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d, t in zip(dataset["rewards"], dataset["terminals"], dataset["timeouts"]):
        ep_ret += float(r)
        ep_len += 1
        if d or t or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def compute_annomaly_score(normal_data: np.ndarray, test_data: np.ndarray, k: int = 10) -> float:
    d = scipy.spatial.distance.cdist(normal_data, [test_data], 'euclidean')
    d = np.partition(d.flatten(),(k-1))
    return - np.log(k) + normal_data.shape[1] * np.log(d[k-1])

def modify_reward(dataset, env_name, max_episode_steps):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d", "maze2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= 1000
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()).detach() * u**2) # added .detach() for weights


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2, exp_order: int = 4,
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        q1, q2 = [], []
        for _ in range(exp_order + 1):
            q1.append(MLP(dims, squeeze_output=True))
            q2.append(MLP(dims, squeeze_output=True))
        self.q1 = nn.ModuleList(q1)
        self.q2 = nn.ModuleList(q2)

    def both(self, state: torch.Tensor, action: torch.Tensor, K: int,) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        q1_val, q2_val = [], []
        for n in range(K + 1):
            q1_val.append(self.q1[n](sa))
            q2_val.append(self.q2[n](sa))
        return torch.stack(q1_val, dim=0), torch.stack(q2_val, dim=0)

    def forward(self, state: torch.Tensor, action: torch.Tensor, K: int) -> torch.Tensor:
        return torch.min(*self.both(state, action, K))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2, exp_order: int = 4,):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        v = []
        for _ in range(exp_order + 1):
            v.append(MLP(dims, squeeze_output=True))
        self.v = nn.ModuleList(v)

    def forward(self, state: torch.Tensor, K: int) -> torch.Tensor:
        out = []
        for n in range(K + 1):
            out.append(self.v[n](state))
        return torch.stack(out, dim=0)
    

class SeqImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount_base: float = 0.99,
        discount: float = 0.999,
        tau: float = 0.005,
        exp_order: int = 4,
        lr_warm_up_period: int = 0,
        training_stage_periods = 100000,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps - lr_warm_up_period)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount_base = discount_base
        self.discount = discount
        self.tau = tau
        self.exp_order = exp_order
        self.lr_warm_up_period = lr_warm_up_period
        self.training_stage_periods = (np.arange(exp_order) + 1) * training_stage_periods
        
        self.total_it = 0
        self.device = device
    

    def _update_critic(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        K: int,
        log_dict: Dict,
    ):
        self.critic_optimizer.zero_grad()
        
        # compute loss for Q
        targets = rewards + (1.0 - torch.stack([terminals for _ in range(next_v.shape[0])], dim=0).float()) * self.discount_base * next_v.detach()
        qs = self.qf.both(observations, actions, K)
        q_loss = sum(F.mse_loss(q, targets) for q in qs)
        log_dict["q_loss"] = q_loss.item()

        # compute loss for V
        with torch.no_grad():
            target_q = self.q_target(observations, actions, K)
        v = self.vf(observations, K)
        adv = target_q.detach() - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        
        # update Q and V
        loss = (q_loss + v_loss) * (K+1)
        loss.backward()
        self.critic_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

        return torch.sum(adv.detach(), dim=0)


    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        self.actor_optimizer.zero_grad()
        
        exp_adv = torch.exp(self.beta * adv).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        policy_loss.backward()
        self.actor_optimizer.step()

    
    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        K = np.sum(self.total_it > self.training_stage_periods) 
        with torch.no_grad():
            next_v = self.vf(next_observations, K)
        # Update critic (Q and V functions)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        rewards = torch.cat((rewards.unsqueeze(0), (self.discount - self.discount_base) * next_v[:-1]), dim=0)
        adv = self._update_critic(next_v, observations, actions, rewards, dones, K, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)
        if self.total_it > self.lr_warm_up_period:
            self.actor_lr_schedule.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.q_target = copy.deepcopy(self.qf)
        self.vf.load_state_dict(state_dict["vf"])
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = env.get_dataset()

    if config.normalize_reward:
        modify_reward(dataset, config.env, max_episode_steps=env._max_episode_steps)

    dataset = d4rl.qlearning_dataset(env, dataset)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    critic_optimizer = torch.optim.Adam(list(v_network.parameters()) + list(q_network.parameters()), lr=config.critic_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "v_network": v_network,
        "critic_optimizer": critic_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # SeqIQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "discount_base": config.discount_base,
        "exp_order": config.exp_order,
        "lr_warm_up_period": config.lr_warm_up_period,
        "training_stage_periods": config.training_stage_periods,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training SeqIQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = SeqImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    last_eval_t = config.max_timesteps // config.eval_freq * config.eval_freq - 1
    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, trajectory, anomaly_score = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=100 if (t == last_eval_t) else 10,
                seed=config.seed,
                rec_trajectory=False,
                rec_annomaly_score=(t == last_eval_t),
                dataset=dataset,
            )
            # eval_scores, trajectory = eval_actor(
            #     env,
            #     actor,
            #     device=config.device,
            #     n_episodes=config.n_episodes,
            #     seed=config.seed,
            # )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {100 if (t == last_eval_t) else 10} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}, anomaly score: {np.mean(anomaly_score)}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score,
                 "anomaly_score_mean": np.mean(anomaly_score),
                }, step=trainer.total_it
            )


if __name__ == "__main__":
    train()
 