import torch
import numpy as np
import itertools
from networks.MIX import HyperNet
from networks.Q_net import Q_net
import copy
from utils.ReplayBuffer import ReplayBuffer
from utils.preference_pool import Preference

class Agents():
    def __init__(self, args):
        self.args = args
        self.policy = [Q_net(args) for _ in range(args.n_agents)]
        self.hyperNet = HyperNet(args)
        self.policy_target = [copy.deepcopy(p) for p in self.policy]
        self.hyperNet_target = copy.deepcopy(self.hyperNet)
        self.replayBuffer = ReplayBuffer(args)
        self.preference_pool = Preference(args)
        policy_param = [policy.parameters() for policy in self.policy]
        self.optim = torch.optim.Adam(itertools.chain(*policy_param,
                                                      self.hyperNet.parameters()), lr=self.args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1000, gamma=0.95, last_epoch=-1)

    def choose_action(self, obs, preference):
        obs = np.array(obs).transpose((1,0,2))
        preference = np.array(preference).transpose((1,0,2))
        act = np.array([self.policy[i].choose_action(obs[i], preference[i])
               for i in range(self.args.n_agents)])

        return act.transpose((2,0,1))

    def learn(self):
        def combine(obs, pref):
            ow = []
            n_pref = len(pref)
            for w in range(n_pref):
                ow.append(torch.cat([obs, pref[w]]).unsqueeze(0))
            ow = torch.cat(ow, dim=0)
            return ow

        sample = self.replayBuffer.sample(self.args.batch_size)
        batch_w = self.preference_pool.sample(32, train=True)
        obs = sample["obs"]
        obs_ = sample["next_obs"]
        act = sample["act"]
        rew = sample["rew"]
        state = sample["state"]
        state_ = sample["next_state"]
        Q_ = []
        for i in range(self.args.batch_size):
            Q_.append([])
            for j in range(32):
                Q_[i].append(torch.cat([self.policy_target[i].get_target_q(combine(obs_[a][i], batch_w[a]), batch_w[a][j])
                      for a in range(self.args.n_agents)]).unsqueeze(0))
            Q_[i] = torch.cat(Q_[i], dim=0).unsqueeze(0)
        Q_ = torch.cat(Q_, dim=0).permute(1, 0, 2)
        Q_ = Q_.reshape((-1, Q_.shape[-1]))
        # Q_tot = self.hyperNet()
        obs = [torch.cat([obs[i] for _ in range(32)]) for i in range(self.args.n_agents)]
        w = copy.deepcopy(batch_w[0])
        batch_w = [batch_w[i].data.cpu().numpy().repeat(self.args.batch_size, axis=0) for i in range(self.args.n_agents)]
        Q = torch.cat([self.policy[i].get_q(obs[i], batch_w[i], act[i]) for i in range(self.args.n_agents)], dim=-1)
        temp = Q.data.numpy()
        Q_tot = self.hyperNet.get_Q_tot(state, w, Q)
        Q_tot_target = self.hyperNet_target.get_Q_tot(state_, w, Q_).detach()
        rew = rew.unsqueeze(0).repeat([32, 1, 1]).view(-1, self.args.n_obj)
        loss = self.loss_func(Q_tot, Q_tot_target,rew, w)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.lr_scheduler.step()

    def loss_func(self, Q, Q_target, R, w):
        temp = Q.data.numpy()
        y = R + Q_target
        w = w.repeat([2, 1]).view(-1, self.args.n_obj)
        temp = y - Q
        La = torch.norm(y - Q, p=2, dim=-1).mean()
        wy = torch.bmm(w.unsqueeze(1), y.unsqueeze(-1))
        wq  =torch.bmm(w.unsqueeze(1), Q.unsqueeze(-1))
        Lb = torch.abs(wy - wq).mean()
        loss = La + Lb
        return loss

    def push(self, traj):
        self.replayBuffer.push(traj["obs"], traj["acts"], traj["rew"], traj["next_obs"],
                               traj["done"], traj["state"], traj["next_state"], traj["pref"])

    def convert_type(self, input):
        if not isinstance(input, torch.Tensor):
            input = torch.Tensor(input)
        if input.device != torch.device(self.args.device):
            input = input.to(self.args.device)

        return input