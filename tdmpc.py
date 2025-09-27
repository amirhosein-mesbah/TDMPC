import os
import random
import time
from dataclasses import dataclass


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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = 'Meta-World/MT1'
    env_name: str = 'reach-v3'
    num_envs: int = 4
    """the number of parallel game environments"""
    seed: int = 42
    
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


if __name__=='__main__':
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.env_name, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    
    
    print('finished')
