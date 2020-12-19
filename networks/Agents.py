import torch
import numpy as np
from networks.MIX import HyperNet
from networks.Q_net import Q_net
import copy
from utils.ReplayBuffer import ReplayBuffer
from utils.preference_pool import Preference

class Agents():
    def __init__(self, args):
        self.args = args
        self.policy = [Q_net(args) for a in range(args.n_agents)]
        self.hyperNet = HyperNet(args)
        self.policy_target = [copy.deepcopy(p) for p in self.policy]
        self.hyperNet_target = copy.deepcopy(self.hyperNet)
        self.replayBuffer = ReplayBuffer(args)
        self.preference_pool = Preference(args)

    def choose_action(self, obs, preference):
        obs = np.array(obs).transpose((1,0,2))
        preference = np.array(preference).transpose((1,0,2))
        # print("o,p", obs, preference)
        act = np.array([self.policy[i].choose_action(obs[i], preference[i])
               for i in range(self.args.n_agents)])

        return act.transpose((2,0,1))

    def learn(self):
        sample = self.replayBuffer.sample(self.args.batch_size)
        batch_w = self.preference_pool.sample(32, train=True)
        obs = sample["obs"]
        obs_ = sample["next_obs"]
        # for b in
        Q = [self.policy[i]()]
        Q_tot = self.hyperNet()
        raise NotImplemented

    def push(self, traj):
        # print("traj:", traj)
        # # traj["obs"] =
        # # traj["acts"] = np.vstack(traj["acts"][:])
        # print("traj[\"acts\"]:", traj["acts"])
        # # traj[]
        self.replayBuffer.push(traj["obs"], traj["acts"], traj["rew"], traj["next_obs"],
                               traj["done"], traj["state"], traj["next_state"], traj["pref"])

    def convert_type(self, input):
        if not isinstance(input, torch.Tensor):
            input = torch.Tensor(input)
        if input.device != torch.device(self.args.device):
            input = input.to(self.args.device)

        return input