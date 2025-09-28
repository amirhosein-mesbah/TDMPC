import os
import random
import time
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import List, Union
from copy import deepcopy


import gymnasium as gym
import metaworld
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


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


def make_env(env_id, env_name, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, env_name=env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, env_name=env_name)
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

# Named tuple to represent a single transition.
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done'))

# A list of Transitions represents a full episode.
Episode = List[Transition]

class EpisodicReplayMemory:
    """
    An episodic replay memory.

    Args:
        capacity (int): Maximum number of episodes the buffer can hold.
        device (Union[str, torch.device]): The device to store the data on.
    """
    def __init__(self, capacity: int, device: Union[str, torch.device] = 'cpu'):
        self.memory = deque([], maxlen=capacity)
        self.device = torch.device(device)
        
        # Temporary buffer for the single, ongoing episode
        self._current_episode: List[Transition] = []

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, done: bool):
        """
        Adds a single transition to the current episode. If the episode is 'done',
        it commits the full episode to memory.

        Args:
            state: The current state (numpy array).
            action: The action taken (numpy array).
            reward: The reward received (float).
            done: A boolean indicating episode termination.
        """
        # Convert data to tensors on the correct device. Note: rewards and dones are scalars.
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action_tensor = torch.as_tensor(action).to(self.device)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32).to(self.device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32).to(self.device)

        transition = Transition(state_tensor, action_tensor, reward_tensor, done_tensor)
        self._current_episode.append(transition)

        if done:
            # Add a copy of the completed episode to the main memory
            self.memory.append(list(self._current_episode))
            # Clear the temporary buffer to start a new episode
            self._current_episode.clear()

    def sample(self, batch_size: int) -> List[Episode]:
        """
        Samples a batch of episodes from the replay memory.
        """
        if not self.memory:
            raise ValueError("Cannot sample from an empty buffer.")
        
        actual_batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, actual_batch_size)

    def __len__(self) -> int:
        return len(self.memory)


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
        

    def plan(self):
        pass
    
    def update(self):
        pass
    

if __name__=='__main__':
    
    args = tyro.cli(Args)
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
        [make_env(args.env_id, args.env_name, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    # define agent
    agent = TDMPC()
    
    # define a replay buffer
    buffer = EpisodicReplayMemory(args.buffer_cap, device)
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
 

    num_iterations = int((args.total_steps + args.episode_length) / args.episode_length)
    for iteration in range(1, num_iterations):
        
        # collect trajectories
        obs, _ = envs.reset(seed=args.seed)
        done  = False
        episode_len = 0
        while not done:
            with torch.no_grad():
                is_first_step = (episode_len == 0)
                action = agent.plan(obs, step = iteration, t0 = is_first_step)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions=[action.cpu().numpy()])
            
            done = terminations[0] or truncations[0]
            buffer.add(obs[0], action, rewards[0], next_obs[0], done)
            obs = next_obs
            global_step += 1
            episode_len += 1
            
        print(f"global_step={global_step}, episode_length={episode_len}, buffer_episodes={len(buffer)}")
            
        
        
