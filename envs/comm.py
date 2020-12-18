import argparse
import numpy as np
import random
import torch
from gym.spaces import Box, Discrete
NUM_UE = 10
NUM_Channel = 5
LAMBDA = 3
NEW_TASK_MEAN = 500000000
NEW_TASK_VAR = 100000
B = 1000000000/NUM_UE# 每个用户的信道带宽
var_noise = 10**(-5)
DELTA_T = 0.1
T_UNIT = 0.01


class CommEnv():
    def __init__(self, args=None ,eval=False):
        args = args
        self.args = args
        self.num_user = args.num_user
        self.fe = args.fe
        self.fc = args.fc
        self.alpha = args.alpha
        self.beta = args.beta
        self.T_max = args.T_max
        self.discrete = args.discrete
        if self.discrete:
            self.ACTIONS = [i * 0.1 for i in range(11)]
        self.eval = eval

    def init_channel_matrix(self):
        UE_Channel_matrix = np.zeros((NUM_Channel, self.num_user))
        for j in range(self.num_user):
            if j < NUM_Channel:
                UE_Channel_matrix[j, j] = 1
            else:
                UE_Channel_matrix[int(j - NUM_Channel), j] = 1
        UE_Channel_matrix = np.array(UE_Channel_matrix)

        return UE_Channel_matrix

    def reset(self):
        self.n_step = 0
        self.ep_r = np.array([0, 0])
        self.task_remain = np.zeros(self.num_user)
        self.UE_Channel_matrix = self.init_channel_matrix()
        self.create_new_task(DELTA_T)
        self.He = abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel, self.num_user) + 1j * np.random.randn(NUM_Channel, self.num_user)))  # 边缘
        self.Hc = 0.1 * abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel, self.num_user) + 1j * np.random.randn(NUM_Channel, self.num_user)))  # 云
        obs = torch.Tensor(list(self.task_remain[0].reshape(-1)) + list(self.He.T[0].reshape(-1)) + list(self.Hc.T[0].reshape(-1)))
        # obs = [np.concatenate([self.task_remain[i].reshape(-1), self.He.T[i].reshape(-1), self.Hc.T[i].reshape(-1)]) for i in range(self.num_user)]
        return obs

    def sum_rate(self, UE_Channel_matrix, He, Hc, pe, pc, B, var_noise):
        pe = np.array(pe)
        pc = np.array(pc)
        He = np.array(He)
        Hc = np.array(Hc)
        rate_edge = np.zeros((NUM_Channel,NUM_UE))
        rate_cloud = np.zeros((NUM_Channel,NUM_UE))
        # 边缘网络
        for n in range(NUM_Channel):
            U = np.transpose(np.nonzero(UE_Channel_matrix[n,:]))
            L = len(U)
            for m in range(L):
                if L > 1:
                    Com_H = He[n, U[m]]
                    # 串行干扰消除干扰用户计算
                    I1 = np.zeros(L)
                    I2 = np.zeros(L)
                    for m1 in range(L):  # 依次检验所有用户
                        if Com_H <= He[n, U[m1]]:  # 大信号 包含原信号
                            I1[m1] = He[n, U[m1]] ** 2 * pe[U[m1]]
                        else:
                            I2[m1] = 0.01 * He[n, U[m1]] ** 2 * pe[U[m1]]
                    rate_edge[n,U[m]] = B * np.math.log2(1 + He[n, U[m]] ** 2 * pe[U[m]] / (var_noise + sum(I1[:]) + sum(I2[:]) - He[n, U[m]] ** 2 * pe[U[m]]))
                elif L == 1:
                    rate_edge[n,U[m]] = B * np.math.log2(1 + He[n, U[m]] ** 2 * pe[U[m]] / var_noise)

        # 云网络
        for n in range(NUM_Channel):
            U = np.transpose(np.nonzero(UE_Channel_matrix[n,:]))
            L = len(U)
            for m in range(L):
                if L > 1:
                    Com_H = Hc[n, U[m]]
                    I1 = np.zeros(L)
                    I2 = np.zeros(L)
                    for m1 in range(L):
                        if Com_H <= Hc[n, U[m1]]:
                            I1[m1] = Hc[n, U[m1]] ** 2 * pc[U[m1]]
                        else:
                            I2[m1] = 0.01 * Hc[n, U[m1]] ** 2 * pc[U[m1]]
                    rate_cloud[n,U[m]] =  B * np.math.log2(1 + Hc[n, U[m]] ** 2 * pc[U[m]] / (var_noise + sum(I1[:]) + sum(I2[:]) - Hc[n, U[m]] ** 2 * pc[U[m]]))
                elif L == 1:
                    rate_cloud[n,U[m]] =  B * np.math.log2(1 + Hc[n, U[m]] ** 2 * pc[U[m]] / var_noise)

        # 云网络
        for n in range(NUM_Channel):
            U = np.transpose(np.nonzero(UE_Channel_matrix[n,:]))
            L = len(U)
            for m in range(L):
                if L > 1:
                    Com_H = Hc[n, U[m]]
                    I1 = np.zeros(L)
                    I2 = np.zeros(L)
                    for m1 in range(L):
                        if Com_H <= Hc[n, U[m1]]:
                            I1[m1] = Hc[n, U[m1]] ** 2 * pc[U[m1]]
                        else:
                            I2[m1] = 0.01 * Hc[n, U[m1]] ** 2 * pc[U[m1]]
                    rate_cloud[n,U[m]] =  B * np.math.log2(1 + Hc[n, U[m]] ** 2 * pc[U[m]] / (var_noise + sum(I1[:]) + sum(I2[:]) - Hc[n, U[m]] ** 2 * pc[U[m]]))
                elif L == 1:
                    rate_cloud[n,U[m]] =  B * np.math.log2(1 + Hc[n, U[m]] ** 2 * pc[U[m]] / var_noise)

        return rate_edge.transpose(), rate_cloud.transpose()

    def compute_reward(self, UE_Channel_matrix, task_coef, pe, pc, task_current):
        task_coef = [1 - task_coef[i] for i in range(self.num_user)]
        E_off = np.zeros([self.num_user])
        E_exe = np.zeros([self.num_user])
        E = np.zeros([self.num_user])
        T_off = np.zeros([self.num_user])
        T_exe = np.zeros([self.num_user])
        T = np.zeros([self.num_user])
        reward = np.zeros([self.num_user])

        rate_edge, rate_cloud = self.sum_rate(UE_Channel_matrix, self.He, self.Hc, pe, pc, B, var_noise)
        rate_edge = rate_edge.transpose() + 0.1e-5
        rate_cloud = rate_cloud.transpose() + 0.1e-5
        for j in range(self.num_user):
            i = np.transpose(np.nonzero(UE_Channel_matrix[:,j]))[0][0]
            E_off[j] = (pe[j] * task_coef[j] * task_current[j]) / rate_edge[i,j] + (pc[j] * (1 - task_coef[j]) * task_current[j]) / rate_cloud[i,j]
            T_off[j] = (task_coef[j] * task_current[j]) / rate_edge[i, j] + ((1 - task_coef[j]) * task_current[j]) / rate_cloud[i,j]
            E_exe[j] = self.beta * (self.alpha * task_coef[j] * task_current[j] * self.fe ** 2 + self.alpha * (1- task_coef[j]) * task_current[j]* self.fc ** 2)
            T_exe[j] = (self.alpha * task_coef[j] * task_current[j]) / self.fe + (self.alpha * (1 - task_coef[j]) * task_current[j]) / self.fc
            E[j] = E_off[j] + E_exe[j]
            T[j] = T_off[j] + T_exe[j]
            if E[j] == 0:
                reward[j] = 0
            else:
                reward[j] = 1/E[j]

        print("T:", T)
        return reward, E, T

    def create_new_task(self, delta_t):
        n = int(delta_t//T_UNIT)
        for u in range(self.num_user):
            for i in range(n):
                task_num = np.random.poisson(lam=self.args.lam, size=1)[0]
                for i in range(task_num):
                    new_task = random.normalvariate(self.args.mean_normal, self.args.var_normal)
                    self.task_remain[u] += new_task

    def step(self, actions, delta_t=None):
        self.n_step += 1
        if delta_t == None:
            delta_t = DELTA_T
        delta_t += self.args.processing_period

        x = [self.ACTIONS[i[0]] for i in actions]
        Pe = [self.ACTIONS[i[1]] for i in actions]
        Pc = [self.ACTIONS[i[2]] for i in actions]
        rate_edge, rate_cloud = self.sum_rate(self.UE_Channel_matrix, self.He, self.Hc, Pe, Pc, B, var_noise)
        offloaded_data = np.zeros_like(self.task_remain)
        for i in range(self.num_user):
            offloaded_data[i] = (x[i] * rate_cloud[i].sum() + (1 - x[i]) * rate_edge[i].sum()) * self.args.processing_period
            self.task_remain[i] -= offloaded_data[i]
            if self.task_remain[i] < 0:
                offloaded_data[i] += self.task_remain[i]
                self.task_remain[i] = 0

        obj_e, _, _ = self.compute_reward(self.UE_Channel_matrix, x, Pe, Pc, offloaded_data)
        reward = np.array([offloaded_data[0], obj_e[0]])
        print("reward:", reward)
        print("task remain before new:", self.task_remain)
        self.ep_r = self.ep_r + reward
        self.create_new_task(delta_t)
        print("task remain after new:", self.task_remain)
        print("----------------------------------------------------------------------------------------")
        self.n_step += 1
        self.He = abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel, self.num_user) + 1j * np.random.randn(NUM_Channel, self.num_user)))  # 边缘
        self.Hc = 0.1 * abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel, self.num_user) + 1j * np.random.randn(NUM_Channel, self.num_user)))  # 云
        obs = np.array(list(self.task_remain[0].reshape(-1)) + list(self.He.T[0].reshape(-1)) + list(self.Hc.T[0].reshape(-1)))
        # obs = [np.concatenate([self.task_remain[i].reshape(-1), self.He.T[i].reshape(-1), self.Hc.T[i].reshape(-1)]) for i in range(self.num_user)]
        return obs, reward, False, {"obj":reward, "episode":{"l":self.n_step, "r":reward}, "obj_raw":None}


    @property
    def observation_space(self):
        if self.num_user > 1:
            return [Box(low=-float("inf"), high=float("inf"), shape=(51,)) for i in range(self.num_user)]
        else:
            return Box(low=-float("inf"), high=float("inf"), shape=(51,))

    @property
    def action_space(self):
        if self.num_user > 1:
            return [Box(low=0, high=1, shape=(3,)) for i in range(self.num_user)]
        else:
            return Box(low=0, high=1, shape=(3,))



def get_args():
    parser = argparse.ArgumentParser(description="computation offloading environment")
    parser.add_argument('--fe', default=10**14)
    parser.add_argument('--fc', default=10**15)
    parser.add_argument('--alpha', default=10**8)
    parser.add_argument('--beta', default=10**(-46))
    parser.add_argument('--T_max', default=8)
    parser.add_argument('--lam', default=100)
    parser.add_argument('--mean_normal', default=100000)
    parser.add_argument('--var_normal', default=10000)
    parser.add_argument('--num_user', default=10)
    parser.add_argument('--processing_period', default=0.01)
    parser.add_argument('--discrete', default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    env = CommEnv(args)
    env.reset()
    for i in range(1000):
        action = [[np.random.randint(0, 10) for _ in range(3)] for _ in range(args.num_user)]
        # print("action:", action)
        # state, reward, done, info = env.step(action, 0.001)
        # print(info)
        print(env.step(action, 0.001))
