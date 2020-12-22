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
import tensorboardX
import copy

def cal_average_score(window, step, writer, tag="scalarized value"):
    r = copy.deepcopy(window['r'])
    w = copy.deepcopy(window['w'])
    l = len(r)
    scalarized = sum([r[i].dot(w[i]) for i in range(l)]) / l
    writer.add_scalar(tag, scalarized, step)

def average_reward(window):
    r = copy.deepcopy(window['r'])
    w = copy.deepcopy(window['w'])
    l = len(r)
    scalarized = sum([r[i].dot(w[i]) for i in range(l)]) / l
    # writer.add_scalar(tag, scalarized, step)
    return scalarized

def run():
    window = {"r":[], "w":[]}
    agent_args = get_args()
    env_args = get_env_args()
    agent_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", agent_args.device)
    writer = tensorboardX.SummaryWriter(env_args.log_dir)
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
    update_step = 0
    record_step = 0
    for ep in range(agent_args.epoches // agent_args.n_threads):
        print('epoch:{0}--------------------------------------------'.format(ep))
        obs = env.reset()
        state = env.get_state()
        w = preferences.sample(1, require_tensor=False)[0]
        tot_rew = 0
        for step in range(200):
            if agent_args.epsilon > 0.05:
                if total_step % 1000 == 0:
                    agent_args.epsilon = agent_args.epsilon * 0.8
            else:
                agent_args.epsilon = 0.05
            acts = agents.choose_action(obs, w, agent_args.epsilon)
            obs_, rew, done, info = env.step(acts)
            for i in range(len(rew)):
                window['r'].append(rew[i])
                window['w'].append(np.array(w[i][0]))
                if len(window) > agent_args.n_threads:
                    window['r'].pop(0)
                    window['w'].pop(0)
            # if record_step % 1 == 0:
            #     cal_average_score(window, total_step, writer)
            tot_rew += average_reward(window)
            # print("reward:", rew)
            # print('preference:', w)
            state_ = env.get_state()
            pref = [w[0][0]]
            traj = {"obs":obs, "rew":rew, "acts":acts, "done":done, "pref":pref,
                    "next_obs":obs_, "state":state , "next_state":state_}
            agents.push(traj)
            state = state_
            obs = obs_
            # print("agents.replayBuffer.__len__():", agents.replayBuffer.__len__())
            # print("agent_args.batch_size:", agent_args.batch_size)
            # print("update_step:", update_step)
            # print("agent_args.update_step:", agent_args.update_step)
            if ((agents.replayBuffer.__len__() >= agent_args.batch_size) and update_step == agent_args.update_step):
                for t in range(agent_args.update_times):
                    print('updating... step:{0}'.format(t))
                    agents.learn()
                print("update finish")
            if update_step == agent_args.update_step:
                update_step = 0
            if total_step % 1000 == 0:
                agents.save_model(total_step)
            agents.update_target()
            # total_step += agent_args.n_threads
            total_step += 1
            update_step += 1
            record_step += 1
        print("ep reward:", tot_rew)
        writer.add_scalar("total reward:", tot_rew, ep)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", default=3)
    parser.add_argument("--gpu", default=False)
    parser.add_argument("--buffer_size", default=30000)
    parser.add_argument("--h_dim", default=128)
    parser.add_argument("--n_obj", default=2)
    parser.add_argument("--hyper_h1", default=128)
    parser.add_argument("--n_threads", default=5)
    parser.add_argument("--batch_size", default=4096)
    parser.add_argument("--epoches", default=1000)
    parser.add_argument('--preference_distribution', default="uniform")
    parser.add_argument('--epsilon', default=0.6)
    parser.add_argument('--norm_rews', default=True)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--update_step', default=200)
    parser.add_argument('--batch_size_p', default=1)
    parser.add_argument('--update_times', default=2)

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
    parser.add_argument('--num_user', default=3)
    parser.add_argument('--processing_period', default=0.01)
    parser.add_argument('--discrete', default=True)
    parser.add_argument("--n_threads", default=5)
    parser.add_argument('--log_dir', default='./log2')

    return parser.parse_args()


if __name__ == '__main__':
    run()