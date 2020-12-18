import torch
import numpy as np
from MIX import HyperNet
from Q_net import Q_net

class Agents():
    def __init__(self, args):
        self.args = args
        self.agents = [Q_net(args) for a in range(args.n_agents)]
        self.hyperNet = HyperNet(args)

    def choose_action(self, obs, preference):
        obs = [self.convert_type(o) for o in obs]
        preference = self.convert_type(preference)
        act = [self.agents[i].choose_action(obs[i], preference)
               for i in range(self.args.n_agents)]

        return act

    def learn(self):
        raise NotImplemented

    def convert_type(self, input):
        if not isinstance(input, torch.Tensor):
            input = torch.Tensor(input)
        if input.device() != torch.device(self.args.device):
            input = input.to(self.args.device)

        return input