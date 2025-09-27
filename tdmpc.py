import os
import random
import time
from dataclasses import dataclass


import gymnasium as gym
import metaworld
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    pass



if __name__=='__main__':
    print('test')
    env = gym.make('Meta-World/MT1', env_name='reach-v3')

    obs = env.reset()
    a = env.action_space.sample()
    next_obs, reward, terminate, truncate, info = env.step(a)