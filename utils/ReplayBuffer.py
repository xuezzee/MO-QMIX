import torch
import numpy as np

class ReplayBuffer():
    def __init__(self, args):
        self.args = args
        self.to_gpu = args.gpu
        self.obs_buffs = [np.zeros((args.buffer_size, args.o_dim)) for _ in range(args.n_agents)]
        self.act_buffs = [np.zeros((args.buffer_size, args.a_dim)) for _ in range(args.n_agents)]
        self.rew_buffs = [np.zeros((args.buffer_size, args.n_obj)) for _ in range(args.n_agents)]
        self.next_obs_buffs = [np.zeros((args.buffer_size, args.o_dim)) for _ in range(args.n_agents)]
        self.done_buffs = [np.zeros((args.buffer_size, 1)) for _ in range(args.n_agents)]
        self.state_buffs = [np.zeros((args.buffer_size, args.s_dim))]
        self.next_state_buffs = [np.zeros((args.buffer_size, args.s_dim))]

        self.filled_i = 0
        self.curr_i = 0

    def __len__(self):
        return self.filled_i

    def push(self, obs, act, rew, next_obs, dones, state, next_state):
        nentries = obs.shape[0]
        if self.curr_i + nentries > self.args.buffer_size:
            rollover = self.args.buffer_size - self.curr_i
            for a in range(self.args.n_agents):
                self.obs_buffs[a] = np.roll(self.obs_buffs[a], rollover, axis=0)
                self.act_buffs[a] = np.roll(self.act_buffs[a], rollover, axis=0)
                self.rew_buffs[a] = np.roll(self.rew_buffs[a], rollover, axis=0)
                self.done_buffs[a] = np.roll(self.done_buffs[a], rollover, axis=0)
                self.state_buffs[a] = np.roll(self.state_buffs[a], rollover, axis=0)
                self.next_obs_buffs[a] = np.roll(self.next_obs_buffs[a], rollover, axis=0)
                self.next_state_buffs[a] = np.roll(self.next_state_buffs[a], rollover, axis=0)

        for a in range(self.args.n_agents):
            self.obs_buffs[a][self.curr_i:self.curr_i + nentries] = np.vstack(obs[:, a])
            self.act_buffs[a][self.curr_i:self.curr_i + nentries] = act[a]
            self.rew_buffs[a][self.curr_i:self.curr_i + nentries] = rew[:, a]
            self.done_buffs[a][self.curr_i:self.curr_i + nentries] = dones[a]
            self.next_obs_buffs[a][self.curr_i:self.curr_i + nentries] = np.vstack(obs[:, a])
            # self

    def sample(self, batch_size):
        inds = np.random.choice(np.arange(self.filled_i), size=batch_size, replace=False)
        if self.to_gpu:
            cast = lambda x: torch.autograd.Variable(torch.Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: torch.autograd.Variable(torch.Tensor(x), requires_grad = False).cuda()

        if self.args.norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.args.n_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.args.n_agents)]

        return