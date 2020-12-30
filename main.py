import torch
import argparse
import numpy as np
from envs.comm import CommEnv
from Actor_Critic.networks import ACAgents
from DQN.DQN import DQNAgents

def run():
    argsE = get_env_args()
    argsA = get_agent_args()
    env = CommEnv(argsE)
    agent = ACAgents(argsA)
    for i in range(6):
        total_reward = 0
        for ep in range(10):
            obs, info = env.reset()
            obs = [obs[0] for _ in range(argsA.n_agents)]
            ep_reward = 0
            tot_rew = 0
            punish = 0
            action_statistic0 = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
            action_statistic1 = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
            action_statistic2 = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
            for step in range(argsA.steps):
                transition = []
                act = [agent.choose_action(obs)]
                # print("action: {0}".format(act))
                # act = [[act[0][0]] + [7, 7]]
                action_statistic0[act[0][0]] += 1
                action_statistic1[act[0][1]] += 1
                action_statistic2[act[0][2]] += 1
                # print('------------------------------------------------------------')
                # print("act:", act)
                obs_, rew, _, info = env.step(act)
                tot_rew += info['new_task']
                punish += info['punishment']
                # print("reward:",rew)
                # print("observation:", obs_)
                obs_ = [obs_[0] for _ in range(argsA.n_agents)]
                rew = [rew[0]]
                for i in range(argsA.n_agents):
                    # transition.append(
                    #     {"state": obs[i], "action": act[i][0], "next_state": obs_[i], "reward": rew})
                    transition.append(
                        {"state": obs[i], "action": act[0][i], "next_state": obs_[i], "reward": rew})
                obs = obs_
                agent.learn(transition)
                del transition
                ep_reward += rew[0]
            print("episode {0} reward {1} total_reward {2} punishment {3}".format(ep, ep_reward, tot_rew, punish))
            print('action_batch1: ', action_statistic0)
            print('action_batch2: ', action_statistic1)
            print('action_batch3: ', action_statistic2)
            total_reward += tot_rew
        print("total_reward: {0}".format(total_reward))
                # ave += ep_reward
            # print("episode {0} reward {1}".format(ep, ave/50))

def run_DQN():
    argsE = get_env_args()
    argsA = get_agent_args()
    env = CommEnv(argsE)
    agent = DQNAgents(argsA)
    for ep in range(argsA.episodes):
        obs, info = env.reset()
        obs = [obs[0] for _ in range(argsA.n_agents)]
        ep_reward = 0
        tot_rew = 0
        punish = 0
        action_statistic0 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
        action_statistic1 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
        action_statistic2 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
        for step in range(argsA.steps):
            transition = []
            act = [agent.choose_action(obs)]
            action_statistic0[act[0][0]] += 1
            action_statistic1[act[0][1]] += 1
            action_statistic2[act[0][2]] += 1
            obs_, rew, _, info = env.step(act)
            tot_rew += info['new_task']
            punish += info['punishment']
            obs_ = [obs_[0] for _ in range(argsA.n_agents)]
            rew = rew[0]
            for i in range(argsA.n_agents):
                transition.append(
                    {"state": obs[i], "action": act[0][i], "next_state": obs_[i], "reward": rew})
            agent.store_transitions(transition)
            agent.learn()
            obs = obs_
            del transition
            ep_reward += rew
        print("episode {0} reward {1} total_reward {2} punishment {3}".format(ep, ep_reward, tot_rew, punish))
        print('action_batch1: ', action_statistic0)
        print('action_batch2: ', action_statistic1)
        print('action_batch3: ', action_statistic2)
            # total_reward += tot_rew

def get_agent_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_dim', default=3)
    parser.add_argument('--a_dim', default=11)
    parser.add_argument('--episodes', default=100)
    parser.add_argument('--steps', default=5000)
    parser.add_argument('--n_agents', default=3)
    parser.add_argument('--gamma', default=0.99)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def get_env_args():
    parser = argparse.ArgumentParser(description="computation offloading environment")
    parser.add_argument('--fe', default=10 ** 14)
    parser.add_argument('--fc', default=10 ** 15)
    parser.add_argument('--alpha', default=10 ** 8)
    parser.add_argument('--beta', default=10 ** (-46))
    parser.add_argument('--T_max', default=8)
    parser.add_argument('--lam', default=100)
    parser.add_argument('--mean_normal', default=7000000)
    parser.add_argument('--var_normal', default=100000)
    parser.add_argument('--num_user', default=1)
    parser.add_argument('--processing_period', default=0.05)
    parser.add_argument('--discrete', default=True)
    parser.add_argument("--n_threads", default=1)
    parser.add_argument('--log_dir', default='./logs')

    return parser.parse_args()



if __name__ == '__main__':
    # run()
    run_DQN()