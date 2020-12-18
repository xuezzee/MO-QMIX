import torch
import numpy as np
import argparse
import sys
import os
from networks.Agents import Agents
from utils.ReplayBuffer import ReplayBuffer
from envs.comm import CommEnv
from utils.env_wrapper import EnvWrapper

def main():
    agent_args = get_args()
    env_args = get_env_args()
    env = EnvWrapper(env_args, CommEnv)








def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", default=5)
    parser.add_argument("--gpu", default=False)
    parser.add_argument("--buffer_size", default=50000)
    parser.add_argument("--h_dim", default=128)
    parser.add_argument("--n_obj", default=2)
    parser.add_argument("--hyper_h1", default=64)
    parser.add_argument("--n_threads", default=1)

    return parser.parse_args()

def get_env_args():
    parser = argparse.ArgumentParser(description="computation offloading environment")
    parser.add_argument('--fe', default=10**14)
    parser.add_argument('--fc', default=10**15)
    parser.add_argument('--alpha', default=10**8)
    parser.add_argument('--beta', default=10**(-46))
    parser.add_argument('--T_max', default=8)
    parser.add_argument('--lam', default=100)
    parser.add_argument('--mean_normal', default=100000)
    parser.add_argument('--var_normal', default=10000)
    parser.add_argument('--num_user', default=1)
    parser.add_argument('--processing_period', default=0.01)
    parser.add_argument('--discrete', default=True)
    parser.add_argument("--n_threads", default=1)

    return parser.parse_args()


if __name__ == '__main__':
    main()