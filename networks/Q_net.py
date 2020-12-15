import torch
import numpy
from torch import nn


class Q_net(nn.Module):
    def __init__(self, args):
        super(Q_net, self).__init__()
        self.args = args
        self.Linear = nn.Linear(args.s_dim, self.h_dim)
        self.GRU1 = nn.Linear(args.h_dim, args.h_dim)
        self.GRU2 = nn.GRUCell(args.h_dim, args.h_dim, bias=True)
        self.GRU3 = nn.GRUCell(args.h_dim, args.h_dim, bias=True)
        self.out1 = nn.Linear(args.h_dim, args.a_dim)
        self.out2 = nn.Linear(args.h_dim, args.a_dim)
        self.out3 = nn.Linear(args.h_dim, args.a_dim)

    def forward(self, input):
        input = self.convert_type(input)
        h = self.Linear(input)
        h1 = self.Linear(h)
        h2 = self.GRU2(h, h1)
        h3 = self.GRU3(h, h2)
        out1 = self.out1(h)
        out2 = self.out2(h2)
        out3 = self.out3(h3)

        return [out1, out2, out3]

    def convert_type(self, x):
        if isinstance(x, numpy.ndarray):
            x = torch.Tensor(x)
        if x.device() != torch.device(self.args.device):
            x = x.to(self.args.device)
        return x
