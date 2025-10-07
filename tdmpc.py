import os
import random
import time
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import List, Union
from copy import deepcopy
import operator
import tyro
import re



import gymnasium as gym
import metaworld

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as pyd
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    #----------------------------------------------------------------
    #                   Experiment Management
    #----------------------------------------------------------------
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment."""
    seed: int = 1
    """The random seed for the experiment."""
    torch_deterministic: bool = True
    """If toggled, `torch.backends.cudnn.deterministic=False`."""
    cuda: bool = False
    """If toggled, cuda will not be enabled by default if available."""
    track: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases."""
    wandb_project_name: str = "cleanRL-TDMPC"
    """The W&B project name."""
    wandb_entity: str = None
    """The entity (team) of the W&B project."""
    capture_video: bool = False
    """Whether to capture videos of the agent's performance."""

    #----------------------------------------------------------------
    #                     Environment Setup
    #----------------------------------------------------------------
    env_id: str = 'Meta-World/MT1'
    """The ID of the Meta-World environment suite."""
    env_name: str = 'reach-v3'
    """The name of the specific task in the Meta-World suite."""
    num_envs: int = 1
    """The number of parallel game environments (must be 1 for this implementation)."""
    total_timesteps: int = 1000000
    """Total number of environment steps to train for."""
    action_repeat: int = 2
    """The number of times to repeat each action in the environment, for computational efficiency and temporal abstraction."""
    
    #----------------------------------------------------------------
    #                       Replay Buffer
    #----------------------------------------------------------------
    buffer_episode_capacity: int = 10000
    """The maximum number of complete episodes to store in the replay buffer's main memory."""
    buffer_transition_capacity: int = 1000000
    """The maximum number of transitions to store in the SumTree for prioritized sampling. Should be >= total_timesteps."""

    #----------------------------------------------------------------
    #                      Model Architecture
    #----------------------------------------------------------------
    latent_dim: int = 50
    """The dimensionality of the latent state vector 'z' produced by the encoder."""
    mlp_dim: int = 256
    """The number of hidden units in each layer of the MLPs used for the model components."""

    #----------------------------------------------------------------
    #                        Planning (CEM)
    #----------------------------------------------------------------
    warmup_steps: int = 1000
    """Number of steps to take random actions at the beginning to seed the replay buffer with diverse data."""
    horizon: int = 5
    """The number of future steps the planner imagines and optimizes over."""
    num_samples: int = 512
    """The number of purely random action sequences to sample at each CEM iteration."""
    num_elites: int = 64
    """The number of top-performing action sequences to keep (the 'elites') for refining the search distribution."""
    planning_optimization_iterations: int = 6
    """The number of refinement iterations to run the CEM planner for at each time step."""
    mixture_coef: float = 0.05
    """The fraction of planner samples that are guided by the learned policy (exploitation) vs. being random (exploration)."""
    temperature: float = 0.5
    """Controls the 'softness' of the elite selection. Lower temperature = more greedy selection (only the very best matter)."""
    momentum: float = 0.1
    """The momentum coefficient for updating the CEM's mean. Smoothes the search distribution's shift between steps."""
    min_std: float = 0.05
    """Minimum standard deviation for the policy's action distribution to ensure exploration."""
    std_schedule: str = 'linear(1.0, 0.1, 250000)'
    """The schedule for the noise added to the final planned action, decaying over time."""
    horizon_schedule: str = 'linear(5, 5, 250000)'
    """A schedule to potentially increase the planning horizon over time (kept constant in the paper)."""
    
    #----------------------------------------------------------------
    #                       Training / Update
    #----------------------------------------------------------------
    batch_size: int = 256
    """The number of sequences to sample from the replay buffer for each gradient update."""
    lr: float = 1e-4
    """The learning rate for the Adam optimizers."""
    discount: float = 0.99
    """The discount factor for future rewards (gamma) in the TD-target calculation."""
    update_freq: int = 2
    """How often to update the target network (in terms of agent steps)."""
    tau: float = 0.01
    """The coefficient for the soft target network update (EMA)."""
    grad_clip_norm: float = 1000.0
    """The maximum norm for gradients to prevent explosion during backpropagation."""
    rho: float = 0.99
    """The temporal discount factor for the multi-step losses. Gives more weight to predictions for earlier, more certain time-steps."""
    consistency_coef: float = 1.0
    """The weight for the latent state consistency loss in the total loss calculation."""
    reward_coef: float = 1.0
    """The weight for the reward prediction loss in the total loss calculation."""
    value_coef: float = 1.0
    """The weight for the Q-value (TD) loss in the total loss calculation."""
    
    
# Environment Setup
class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, amount: int):
        super().__init__(env)
        self.amount = amount
        assert amount > 0

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for _ in range(self.amount):
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        return next_obs, total_reward, terminated, truncated, info


def make_env(env_id, env_name, idx, capture_video, run_name, action_repeat):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, env_name=env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, env_name=env_name)
        if action_repeat > 1:
            env = ActionRepeatWrapper(env, action_repeat)
        return env

    return thunk

# Helper Functions
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
   
def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)


__REDUCE__ = lambda b: 'mean' if b else 'none'

def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


# prioritized replay buffer
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class ReplayMemory:
    def __init__(self,  args: Args, device: Union[str, torch.device] = 'cpu'):
        self.device = torch.device(device)
        self.args = args
        self.memory = deque([], maxlen=args.buffer_episode_capacity)

        # PER parameters
        self.per_alpha = 0.6 
        self.per_beta = 0.4 
        self.beta_increment = (1.0 - self.per_beta) / args.total_timesteps
        self.max_priority = 1.0
        self.tree = SumTree(args.buffer_transition_capacity)

        self._current_episode_data = []
        
    def add(self, state, action, reward, next_state, done):
        self._current_episode_data.append((state, action, reward, next_state, done))

        if done:
            episode = self._current_episode_data
            self.memory.append(episode)
            self._current_episode_data = []
            for i in range(len(episode) - self.args.horizon + 1):
                pointer = (len(self.memory) - 1, i) 
                self.tree.add(self.max_priority, pointer)
                
    def update_priorities(self, tree_indices, priorities):
        priorities = priorities.detach().cpu().numpy().flatten()
        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
            

    def sample(self, batch_size: int):
        if self.tree.n_entries < batch_size:
            return None

        batch_indices, batch_priorities, batch_pointers = [], [], []
        segment = self.tree.total_priority / batch_size
        self.per_beta = min(1.0, self.per_beta + self.beta_increment)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get_leaf(s)
            batch_indices.append(idx)
            batch_priorities.append(p)
            batch_pointers.append(data)


        sampling_probabilities = np.array(batch_priorities) / self.tree.total_priority
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.per_beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        start_obs_batch, action_batch, reward_batch, next_obs_batch = [], [], [], []
        for ep_idx, start_idx in batch_pointers:
            episode = self.memory[ep_idx]
            sequence_data = episode[start_idx : start_idx + self.args.horizon]
            
            states, actions, rewards, next_states, _ = zip(*sequence_data)

            start_obs_batch.append(torch.from_numpy(states[0]).to(self.device))
            action_batch.append(torch.from_numpy(np.array(actions)).to(self.device))
            reward_batch.append(torch.from_numpy(np.array(rewards)).to(self.device))
            next_obs_batch.append(torch.from_numpy(np.array(next_states)).to(self.device))

        # shape [H, B, dim]
        return (
            torch.stack(start_obs_batch).float(),
            torch.stack(action_batch).permute(1, 0, 2).float(),
            torch.stack(reward_batch).permute(1, 0).unsqueeze(-1).float(),
            torch.stack(next_obs_batch).permute(1, 0, 2).float(),
            batch_indices,
            weights
        )

    def __len__(self) -> int:
        return len(self.memory)


# Helper classes and functions to implement TOLD and TDMPC
class MLP(nn.Module):
    def __init__(self, mlp_input_dim, mlp_hidden_dim, mlp_output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_output_dim))
            
    def forward(self, x):
        return self.net(x)
    
    
class QNet(nn.Module):
    def __init__(self, latent_dim, action_dim, mlp_hidden_dim):
        super().__init__()
        self.net = MLP(latent_dim+action_dim, mlp_hidden_dim, 1)
    
    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x)
   

def _standard_normal(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device)

class TruncatedNormal(pyd.Normal):
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)  


class TDMPC(nn.Module):
    def __init__(self, args: Args, obs_space, action_space, device):
        super().__init__()
        self.args = args
        self.device = device
        
        action_dim = action_space.shape[0]
        obs_dim = obs_space.shape[0]
        
        # TOLD world model
        if len(obs_space.shape) == 3:
            raise NotImplementedError("CNN Encoder not implemented, this script is not for pixel based environments!") 
        else:
            self._encoder = MLP(obs_dim,  args.mlp_dim, args.latent_dim)
            
        self._dynamics = MLP( args.latent_dim + action_dim, args.mlp_dim,  args.latent_dim)
        self._reward = MLP( args.latent_dim+ action_dim, args.mlp_dim, 1)
        self._pi = MLP( args.latent_dim, args.mlp_dim, action_dim)
        self._Q1 = QNet( args.latent_dim, action_dim, args.mlp_dim,)
        self._Q2 = QNet( args.latent_dim, action_dim, args.mlp_dim,)
        
        # orthogonal initialization
        self.apply(orthogonal_init)
        
        self.apply(orthogonal_init)
        self._reward.net[-1].weight.data.fill_(0)
        self._reward.net[-1].bias.data.fill_(0)
        for m in [self._Q1, self._Q2]:
            m.net.net[-1].weight.data.fill_(0)
            m.net.net[-1].bias.data.fill_(0)
            
        self.model_target = deepcopy(self)
        
        # Optimizers
        model_params = list(self._encoder.parameters()) + list(self._dynamics.parameters()) + \
                       list(self._reward.parameters()) + list(self._Q1.parameters()) + list(self._Q2.parameters())
        self.optim = optim.Adam(model_params, lr=self.args.lr)
        self.pi_optim = optim.Adam(self._pi.parameters(), lr=self.args.lr)
        
        self.to(self.device)
        self.model_target.to(self.device)
        self.model_target.eval()
        
    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            for p in m.parameters():
                p.requires_grad_(enable)

    def h(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)
    
    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std_tensor = torch.ones_like(mu) * std
            dist = TruncatedNormal(mu, std_tensor)
            return dist.sample()
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        return self._Q1(z, a), self._Q2(z, a)

    


if __name__=='__main__':
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
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

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.env_name, i, args.capture_video, run_name, args.action_repeat) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    action_dim = envs.single_action_space.shape[0]
    
    agent = TDMPC(args, envs.single_observation_space, envs.single_action_space, device)
    

    buffer = ReplayMemory(args, device)
    
    # TRY NOT TO MODIFY:
    global_step = 0
    success_rate = 0.0
    episode_count = 0.0
    current_rate = 0.0
    start_time = time.time()
 
    while global_step < args.total_timesteps:
        
        # collect trajectories with planning
        obs, _ = envs.reset(seed=args.seed)
        obs = obs[0]
        done  = False
        episode_len = 0
        prev_mean = None
        cumulative_reward = 0
        while not done:
            with torch.no_grad():
                # if agent was in the warm-up stage take fully random actions
                if global_step < args.warmup_steps:
                    action = torch.empty(action_dim, dtype=torch.float32, device=device).uniform_(-1, 1)
                # else select actions with planning
                # sample sequences of actions (with the length of horizon), some of are sample regarding the learned policy
                # evaluate sequences and select top k sequences
                # update the parameters based on top k
                # step action in the environment and save the transition in buffer
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    horizon = int(round(min(args.horizon, linear_schedule(args.horizon_schedule, global_step))))
                    num_pi_trajs = int(args.mixture_coef * args.num_samples) 
                    if num_pi_trajs > 0:
                        pi_actions = torch.empty(args.horizon, num_pi_trajs, action_dim, device=device)
                        z = agent.h(obs_tensor).repeat(num_pi_trajs, 1)
                        for t in range(horizon):
                            pi_actions[t] = agent.pi(z, args.min_std)
                            z, _ = agent.next(z, pi_actions[t])
                        

                    z_cem_start = agent.h(obs_tensor).repeat(args.num_samples+num_pi_trajs, 1)
                    mean = torch.zeros(horizon, action_dim, device=device)
                    std = 2*torch.ones(horizon, action_dim, device=device)
                    if episode_len > 0 and prev_mean is not None:
                        len_to_copy = min(mean.shape[0], prev_mean.shape[0]) - 1
                        if len_to_copy > 0:
                            mean[:len_to_copy] = prev_mean[1 : 1 + len_to_copy]
                            
                    # Iterate CEM
                    for i in range(args.planning_optimization_iterations):
                        actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                            torch.randn(horizon, args.num_samples, action_dim, device=std.device), -1, 1)
                        if num_pi_trajs > 0:
                            actions = torch.cat([actions, pi_actions], dim=1)

                        # Compute elite actions
                        z = z_cem_start.clone()
                        G, discount = 0, 1
                        for t in range(args.horizon):
                            z, reward = agent.next(z, actions[t])
                            G += discount * reward
                            discount *= args.discount
                        G += discount * torch.min(*agent.Q(z, agent.pi(z, args.min_std)))
                        value = G.nan_to_num_(0)
                        elite_idxs = torch.topk(value.squeeze(1), args.num_elites, dim=0).indices
                        elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

                        # Update parameters
                        max_value = elite_value.max(0)[0]
                        score = torch.exp(args.temperature*(elite_value - max_value))
                        score /= score.sum(0)
                        _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
                        _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
                        _std = _std.clamp_(linear_schedule(args.std_schedule, global_step), 2)
                        mean, std = args.momentum * mean + (1 - args.momentum) * _mean, _std
                        

                    score = score.squeeze(1).cpu().numpy()
                    actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
                    prev_mean = mean
                    action = actions[0]
                    action = torch.clamp(action + std[0] * torch.randn(action_dim, device=device), -1, 1)
            
            # apply action to the env
            next_obs, rewards, terminations, truncations, infos = envs.step(actions=[action.cpu().numpy()])
            
            done = terminations[0] or truncations[0]
            buffer.add(obs, action.cpu().numpy(), rewards[0], next_obs[0], done) 
            obs = next_obs[0]
            global_step += 1
            episode_len += 1
            cumulative_reward += rewards[0]
        
        # Logging episode stats
        episode_count += 1
        if infos["success"][0] == 1:
               success_rate += 1  
        current_rate = success_rate/episode_count
        print(f"ep= {episode_count} global_step={global_step}, episode_length={episode_len}, buffer_episodes={len(buffer)}, cumulative reward={cumulative_reward}, success_rate={current_rate}")
        writer.add_scalar("charts/episode_reward", cumulative_reward, global_step)
        writer.add_scalar("charts/episode_length", episode_len, global_step)
        writer.add_scalar("charts/success_rate", current_rate, global_step)
        
        
        # Update TOLD Model
        # sample from buffer
        # calculate the loss and update the TOLD model components
        # update the policy net with its own objective
        if global_step >= args.warmup_steps:
            for i in range(episode_len):
                obs, actions, rewards, next_obses, idx, weights =  sample_data = buffer.sample(args.batch_size)
                agent.optim.zero_grad(set_to_none=True)
                std = linear_schedule(args.std_schedule, global_step)
                agent.train()

                z = agent.h(obs) 
                zs_for_pi = [z.detach()]

                consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
                for t in range(args.horizon):
                    Q1, Q2 = agent.Q(z, actions[t])
                    z_pred, reward_pred = agent.next(z, actions[t])
                    with torch.no_grad():
                        next_obs = next_obses[t]
                        next_z_target = agent.model_target.h(next_obs)
                        next_z = agent.h(next_obs)
                        td_target = rewards[t] + args.discount * torch.min(*agent.model_target.Q(next_z, agent.pi(next_z, args.min_std)))
                
    
                    # Losses
                    rho = (args.rho ** t)
                    consistency_loss += rho * torch.mean(mse(z_pred, next_z_target), dim=1, keepdim=True)
                    reward_loss += rho * mse(reward_pred, rewards[t])
                    value_loss += rho * (mse(Q1, td_target) + mse(Q2, td_target))
                    priority_loss += rho * (l1(Q1, td_target) + l1(Q2, td_target))

                    z = z_pred
                    zs_for_pi.append(z.detach())
                    
                # Optimize model
                total_loss = args.consistency_coef * consistency_loss.clamp(max=1e4) + \
                            args.reward_coef * reward_loss.clamp(max=1e4) + \
                            args.value_coef * value_loss.clamp(max=1e4)
                weighted_loss = (total_loss * weights).mean()
                weighted_loss.register_hook(lambda grad: grad * (1/args.horizon))
                weighted_loss.backward()
                
                model_params = list(agent._encoder.parameters()) + list(agent._dynamics.parameters()) + \
               list(agent._reward.parameters()) + list(agent._Q1.parameters()) + list(agent._Q2.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(model_params, args.grad_clip_norm, error_if_nonfinite=False) 
                agent.optim.step()
                buffer.update_priorities(idx, priority_loss.clamp(max=1e4).detach())
                
                
                # Update policy network
                agent.pi_optim.zero_grad(set_to_none=True)
                agent.track_q_grad(False)

                pi_loss = 0
                for t, z_pi in enumerate(zs_for_pi):
                    a = agent.pi(z_pi, args.min_std)
                    Q = torch.min(*agent.Q(z_pi, a))
                    pi_loss += -Q.mean() * (args.rho ** t)

                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent._pi.parameters(), args.grad_clip_norm)
                agent.pi_optim.step()
                agent.track_q_grad(True)
                
                if global_step % args.update_freq == 0:
                    with torch.no_grad():
                        for p, p_target in zip(agent.parameters(), agent.model_target.parameters()):
                            p_target.data.lerp_(p.data, args.tau)


                agent.eval()
                
            writer.add_scalar("losses/value_loss", float(value_loss.mean().item()), global_step)
            writer.add_scalar("losses/policy_loss", pi_loss.item(), global_step)
            writer.add_scalar("losses/consistency_loss", float(consistency_loss.mean().item()), global_step)
            writer.add_scalar("losses/total_loss", float(total_loss.mean().item()), global_step)
            writer.add_scalar("losses/weighted_loss", float(weighted_loss.mean().item()), global_step)
            writer.add_scalar("losses/reward_loss", float(reward_loss.mean().item()), global_step)
            writer.add_scalar("charts/grad_norm", float(grad_norm), global_step)

 
    envs.close()
    writer.close()
    if args.track:
        wandb.finish()