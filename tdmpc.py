import os
import random
import time
from dataclasses import dataclass
from collections import deque, namedtuple


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
    num_envs: int = 4
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

# Named tuple to represent a single transition in the replay buffer.
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Class for standard experience replay memory.
class ReplayMemory:
    """
    A standard replay memory for storing and sampling transitions.

    Args:
        capacity (int): Maximum number of transitions the buffer can hold.
    """
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)  # Circular buffer to store transitions

    def add(self, states, actions, rewards, next_states, dones):
        """
        Adds a batch of transitions to the replay memory.

        Args:
            states: List of current states.
            actions: List of actions taken.
            rewards: List of rewards received.
            next_states: List of next states.
            dones: List of done flags indicating episode termination.
        """
        num_envs = len(states)
        if not (len(actions) == num_envs and len(next_states) == num_envs and 
                len(rewards) == num_envs and len(dones) == num_envs):
            raise ValueError("All input iterables must have the same length (number of environments)")
        
        transitions_iterable = (Transition(s, a, r, ns, d)
                                 for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones))
        self.memory.extend(transitions_iterable)


    def sample(self, batch_size: int) -> list[Transition]:
        """
        Samples a batch of transitions from the replay memory.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            list[Transition]: A list of sampled transitions.
        """
        if not self.memory: # Check if memory is empty
            raise ValueError("Cannot sample from an empty buffer.")
        if batch_size > len(self.memory):
            # print(f"Warning: Sampling {len(self.memory)} (available) instead of {batch_size}")
            # Or raise error if strict batch_size is needed and buffer not full
            return random.sample(self.memory, len(self.memory))
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Returns the current size of the replay memory.

        Returns:
            int: Number of transitions in the memory.
        """
        return len(self.memory)

class TDMPC:
    def __init__(self):
        pass
    
    def plan(self):
        pass
    
    def

if __name__=='__main__':
    
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

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
    
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.env_name, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    # define agent
    
    # define a replay buffer
    buffer = ReplayMemory(args.buffer_cap)
    
    print('finished')
