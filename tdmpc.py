import os
import random
import time
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import List, Union
from copy import deepcopy
import operator


import gymnasium as gym
import metaworld

import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

__REDUCE__ = lambda b: 'mean' if b else 'none'


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    
    # Algorithm specific arguments
    env_id: str = 'Meta-World/MT1'
    """the id of the environment"""
    env_name: str = 'reach-v3'
    """name of the task for metaworld"""
    num_envs: int = 1
    """the number of parallel game environments"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_cap: int = 1000000
    """capacity of replay buffer"""

    
    capture_video: bool = True
    
    
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
        env = ActionRepeatWrapper(env, action_repeat)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

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



def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))

# Named tuple to represent a single transition.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# A list of Transitions represents a full episode.
Episode = List[Transition]

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
    """
    An episodic replay memory with Prioritized Experience Replay (PER).
    This buffer stores full episodes but samples fixed-length sequences based on priority.
    """
    def __init__(self,  args: Args, device: Union[str, torch.device] = 'cpu'):
        self.device = torch.device(device)
        self.args = args
        self.memory = deque([], maxlen=args.buffer_cap)

        # PER parameters
        self.per_alpha = 0.6 # How much prioritization to use (0=uniform, 1=fully prioritized)
        self.per_beta = 0.4 # Importance-sampling correction, anneals to 1.0
        self.beta_increment = (1.0 - self.per_beta) / args.total_timesteps
        self.max_priority = 1.0
        self.tree_capacity = args.buffer_cap # Total transition capacity for the SumTree
        self.tree = SumTree(self.tree_capacity)

        self._current_episode_data = []
        
    def add(self, state, action, reward, next_state, done):
        """Adds a single transition and commits the episode when done."""
        # Store transition data temporarily
        self._current_episode_data.append((state, action, reward, next_state, done))

        if done:
            episode = self._current_episode_data
            self.memory.append(episode)
            self._current_episode_data = []

            # Add all valid sequence starting points from this episode to the SumTree
            for i in range(len(episode) - self.args.horizon + 1):
                # The data stored in the tree is a pointer to the episode and the start index
                pointer = (len(self.memory) - 1, i) 
                self.tree.add(self.max_priority, pointer)
                
    def update_priorities(self, tree_indices, priorities):
        """Update priorities of sampled transitions."""
        priorities = priorities.detach().cpu().numpy().flatten()
        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
            

    def sample(self, batch_size: int):
        """Samples a batch of sequences using priorities."""
        if self.tree.n_entries < batch_size:
            return None

        # Sample from the SumTree
        batch_indices, batch_priorities, batch_pointers = [], [], []
        segment = self.tree.total_priority / batch_size
        self.per_beta = min(1.0, self.per_beta + self.beta_increment)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get_leaf(s)
            batch_indices.append(idx)
            batch_priorities.append(p)
            batch_pointers.append(data)

        # Calculate importance-sampling weights
        sampling_probabilities = np.array(batch_priorities) / self.tree.total_priority
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.per_beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Retrieve and format the sequences
        start_obs_batch, action_batch, reward_batch, next_obs_batch = [], [], [], []
        for ep_idx, start_idx in batch_pointers:
            episode = self.memory[ep_idx]
            sequence_data = episode[start_idx : start_idx + self.args.horizon]
            
            states, actions, rewards, next_states, _ = zip(*sequence_data)

            # Convert to tensors here
            start_obs_batch.append(torch.from_numpy(states[0]).to(self.device))
            action_batch.append(torch.from_numpy(np.array(actions)).to(self.device))
            reward_batch.append(torch.from_numpy(np.array(rewards)).to(self.device))
            next_obs_batch.append(torch.from_numpy(np.array(next_states)).to(self.device))

        # Stack and permute to match paper's expected format [H, B, dim]
        return (
            torch.stack(start_obs_batch).float(),
            torch.stack(action_batch).permute(1, 0, 2).float(),
            torch.stack(reward_batch).permute(1, 0).unsqueeze(-1).float(),
            torch.stack(next_obs_batch).permute(1, 0, 2).float(),
            batch_indices,
            weights
        )

    def __len__(self) -> int:
        return len(self.memory) # Number of episodes



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
    
class CNNEncoder(nn.Module):
    def __init__(self, input_channels, encoder_hidden_dim, encoder_latent_dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            # Input shape: [N, C, 84, 84]
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Shape: [N, 32, 20, 20]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Shape: [N, 64, 9, 9]
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Shape: [N, 64, 7, 7]
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 84, 84)
            flattened_size = self.conv_net(dummy_input).shape[1]
        
        self.fc_net = nn.Sequential(
            nn.Linear(flattened_size, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_latent_dim),
        )
    
    def forward(self, obs):
        conv_out = self.conv_net(obs)
        latent_vector = self.fc_net(conv_out)
        return latent_vector
    
    
    
class QNet(nn.Module):
    def __init__(self, latent_dim, action_dim, mlp_hidden_dim):
        super().__init__()
        self.net = MLP(latent_dim+action_dim, mlp_hidden_dim, 1)
    
    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x)
   
   
class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, clip=0.3):
        super().__init__(loc, scale)
        self.low = low
        self.high = high
        self.clip = clip

    def sample(self, sample_shape=torch.Size()):
        # Note: The original implementation has a clip parameter, which suggests
        # they might be clipping the sample, not just the distribution range.
        # This is a simplified version.
        x = super().sample(sample_shape)
        return torch.clamp(x, self.low + self.clip, self.high - self.clip)     


class TDMPC(nn.Module):
    """
    The integrated TD-MPC agent, combining the TOLD model and the planning/update logic.
    """
    def __init__(self, args: Args, obs_space, action_space, device):
        super().__init__()
        self.args = args
        self.device = device
        
        # Automatically set action_dim from the environment
        action_dim = action_space.shape[0]
        obs_dim = obs_space.shape[0]
        
        # TOLD world model
        if len(obs_space.shape) == 3:
            print("Using CNN Encoder for pixel-based observations.")
            input_channels = obs_space.shape[0]
            self._encoder = CNNEncoder(input_channels, args.encoder_hidden_dim, args.encoder_latent_dim)
        else:
            print("Using MLP for vector-based encoder.")
            obs_dim = obs_space.shape[0]
            self._encoder = MLP(obs_dim, self.args.mlp_hidden_dim, self.args.encoder_latent_dim)
            
        self._dynamics = MLP(args.encoder_latent_dim + action_dim, args.mlp_hidden_dim, args.encoder_latent_dim)
        self._reward = MLP(args.encoder_latent_dim + action_dim, args.mlp_hidden_dim, 1)
        self._pi = MLP(args.encoder_latent_dim, args.mlp_hidden_dim, args.action_dim)
        self._Q1 = QNet(args.encoder_latent_dim, action_dim, args.mlp_hidden_dim)
        self._Q2 = QNet(args.encoder_latent_dim, action_dim, args.mlp_hidden_dim)
        
        # orthogonal initialization
        self.apply(orthogonal_init)
        
        self.apply(orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            # Access the last layer of the Sequential net inside the module
            m.net[-1].weight.data.fill_(0)
            m.net[-1].bias.data.fill_(0)
            
        # --- TD-MPC Algorithm Components ---
        self.model_target = deepcopy(self)
        
        # Optimizers
        model_params = list(self._encoder.parameters()) + list(self._dynamics.parameters()) + \
                       list(self._reward.parameters()) + list(self._Q1.parameters()) + list(self._Q2.parameters())
        self.optim = torch.optim.Adam(model_params, lr=self.args.lr)
        self.pi_optim = torch.optim.Adam(self._pi.parameters(), lr=self.args.lr)
        
        
        # Planner state
        self._prev_mean = None
        
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
        return self._dynamics(z, a), self._reward(z, a)
    
    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std_tensor = torch.ones_like(mu) * std
            return TruncatedNormal(mu, std_tensor).sample()
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        return self._Q1(z, a), self._Q2(z, a)

    

if __name__=='__main__':
    
    args = tyro.cli(Args)
    args.num_iterations = (args.total_steps + args.episode_length) // args.episode_length
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
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

    # env setup observation space (1, 39) - action space (1, 4)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.env_name, i, args.capture_video, run_name, args.action_repeat) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    action_dim = envs.single_action_space.shape[0][0]
    
    # define agent
    agent = TDMPC(args, envs.observation_space, envs.action_space, device)
    
    # define a replay buffer
    buffer = ReplayMemory(args, device)
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
 
    for iteration in range(1, args.num_iterations+1):
        
        # collect trajectories
        obs, _ = envs.reset(seed=args.seed)
        obs = obs[0]
        done  = False
        episode_len = 0
        prev_mean = None
        while not done:
            with torch.no_grad():
                if global_step < args.warmup_steps:
                    action = torch.empty(action_dim, dtype=torch.float32, device=device).uniform_(-1, 1)
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    horizon = int(min(args.horizon, linear_schedule(args.horizon_schedule, global_step)))
                    num_pi_trajs = int(args.mixture_coef * args.num_samples) # what fraction of the search should be guided by the policy.
                    if num_pi_trajs > 0:
                        pi_actions = torch.empty(args.horizon, num_pi_trajs, action_dim, device=device)
                        z = agent.h(obs_tensor).repeat(num_pi_trajs, 1)
                        for t in range(horizon):
                            pi_actions[t] = agent.pi(z, args.min_std)
                            z, _ = agent.next(z, pi_actions[t])
                        
                    # Initialize state and parameters
                    z_cem_start = agent.h(obs_tensor).repeat(args.num_samples+num_pi_trajs, 1)
                    mean = torch.zeros(horizon, action_dim, device=device)
                    std = 2*torch.ones(horizon, action_dim, device=device)
                    if episode_len > 0 and prev_mean is not None:
                        mean[:-1] = prev_mean[1:]
                            
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
                        
                    # Outputs
                    score = score.squeeze(1).cpu().numpy()
                    actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
                    prev_mean = mean
                    action = actions[0]
                    action = torch.clamp(action + std[0] * torch.randn(action_dim, device=device), -1, 1)
                    
                # is_first_step = (episode_len == 0)
                # action = agent.plan(obs, step = iteration, t0 = is_first_step)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions=[action.cpu().numpy()])
            
            done = terminations[0] or truncations[0]
            buffer.add(obs, action.cpu().numpy(), rewards[0], next_obs[0], done) 
            obs = next_obs[0]
            global_step += 1
            episode_len += 1
            
        print(f"global_step={global_step}, episode_length={episode_len}, buffer_episodes={len(buffer)}")
        
        # Update Modelp
        if global_step >= args.warmup_steps:
            num_updates = args.warmup_steps if global_step == args.warmup_steps else args.episode_length
            for i in range(num_updates):
                obs, actions, rewards, next_obses, tree_indices, weights =  sample_data = buffer.sample(args.batch_size)
                agent.optim.zero_grad(set_to_none=True)
                std = linear_schedule(args.std_schedule, global_step)
                agent.train()

                # Representation
                z = agent.h(obs)
                zs = [z.detach()]

                consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
                for t in range(args.horizon):
                    # Predictions
                    Q1, Q2 = agent.Q(z, action[t])
                    z, reward_pred = agent.next(z, action[t])
                    with torch.no_grad():
                        next_obs = next_obses[t]
                        next_z = agent.model_target.h(next_obs)
                        next_z = agent.h(next_obs)
                        td_target = reward + args.discount * torch.min(*agent.model_target.Q(next_z, agent.pi(next_z, args.min_std)))
                
                    zs.append(z.detach())

                    # Losses
                    rho = (args.rho ** t)
                    consistency_loss += rho * torch.mean(mse(z, next_z), dim=1, keepdim=True)
                    reward_loss += rho * mse(reward_pred, reward[t])
                    value_loss += rho * (mse(Q1, td_target) + mse(Q2, td_target))
                    priority_loss += rho * (l1(Q1, td_target) + l1(Q2, td_target))
                    
                
                # Optimize model
                total_loss = args.consistency_coef * consistency_loss.clamp(max=1e4) + \
                            args.reward_coef * reward_loss.clamp(max=1e4) + \
                            args.value_coef * value_loss.clamp(max=1e4)
                weighted_loss = (total_loss.squeeze(1) * weights).mean()
                weighted_loss.register_hook(lambda grad: grad * (1/args.horizon))
                weighted_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip_norm, error_if_nonfinite=False)
                agent.optim.step()
                buffer.update_priorities(tree_indices, priority_loss.clamp(max=1e4).detach())
                
                
                # Update policy + target network
                agent.pi_optim.zero_grad(set_to_none=True)
                agent.track_q_grad(False)

                # Loss is a weighted sum of Q-values
                pi_loss = 0
                for t,z in enumerate(zs):
                    a = agent.pi(z, args.min_std)
                    Q = torch.min(*agent.Q(z, a))
                    pi_loss += -Q.mean() * (args.rho ** t)

                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent._pi.parameters(), args.grad_clip_norm, error_if_nonfinite=False)
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
                
            
        # Logging
        episode_idx += 1
		env_step = int(global_step*args.action_repeat)
        
        
        # # Eval Agent
        # if global_step % args.eval_freq == 0:
        #     """Evaluate a trained agent and optionally save a video."""
	    #     episode_rewards = []
	    #     for i in range(num_episodes):
		#     obs, _,  done, ep_reward, t = envs.reset(), False, 0, 0
		#     if video: video.init(env, enabled=(i==0))
		# while not done:
		# 	action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
		# 	obs, reward, done, _ = env.step(action.cpu().numpy())
		# 	ep_reward += reward
		# 	if video: video.record(env)
		# 	t += 1
		# episode_rewards.append(ep_reward)
        
        
