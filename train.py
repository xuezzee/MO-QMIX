import torch
import numpy as np
import argparse
import sys
import os
from networks.Agents import Agents
from utils.ReplayBuffer import ReplayBuffer
from envs.comm import CommEnv
from utils.env_wrapper import EnvWrapper, EnvWrapper_sigle
from utils.preference_pool import Preference

def run():
    agent_args = get_args()
    env_args = get_env_args()
    agent_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if agent_args.n_threads == 1:
        env = EnvWrapper_sigle(env_args, CommEnv)
    else:
        env = EnvWrapper(env_args, CommEnv)
    agent_args.o_dim = env.observation_space.shape[0]
    agent_args.s_dim = env.state_space.shape[0]
    agent_args.a_dim = env.action_space.n

    agents = Agents(agent_args)
    preferences = Preference(agent_args)
    total_step = 1
    for ep in range(agent_args.epoches):
        obs = env.reset()
        state = env.get_state()
        w = preferences.sample(1, require_tensor=False)[0]
        for step in range(1000):
            # print(step)
            acts = agents.choose_action(obs, w)
            obs_, rew, done, info = env.step(acts)
            state_ = env.get_state()
            pref = [w[0][0]]
            traj = {"obs":obs, "rew":rew, "acts":acts, "done":done, "pref":pref,
                    "next_obs":obs_, "state":state , "next_state":state_}
            agents.push(traj)
            state = state_
            obs = obs_
            print(agents.replayBuffer.__len__())
            if ((agents.replayBuffer.__len__() >= agent_args.batch_size)):
                agents.learn()
            total_step += agent_args.n_threads


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", default=5)
    parser.add_argument("--gpu", default=False)
    parser.add_argument("--buffer_size", default=50000)
    parser.add_argument("--h_dim", default=128)
    parser.add_argument("--n_obj", default=2)
    parser.add_argument("--hyper_h1", default=64)
    parser.add_argument("--n_threads", default=1)
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--epoches", default=1000)
    parser.add_argument('--preference_distribution', default="uniform")
    parser.add_argument('--epsilon', default=0.9)
    parser.add_argument('--norm_rews', default=True)
    parser.add_argument('--learning_rate', default=0.001)

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
    parser.add_argument('--num_user', default=5)
    parser.add_argument('--processing_period', default=0.01)
    parser.add_argument('--discrete', default=True)
    parser.add_argument("--n_threads", default=1)

    return parser.parse_args()


if __name__ == '__main__':
    run()