import torch
import numpy as np
from torch import nn
import random


class Q_net(nn.Module):
    def __init__(self, args):
        super(Q_net, self).__init__()
        self.args = args
        self.Linear = nn.Linear(args.o_dim+args.n_obj, args.h_dim)
        self.GRU1 = nn.Linear(args.h_dim, args.h_dim)
        self.GRU2 = nn.GRUCell(args.h_dim, args.h_dim, bias=True)
        self.GRU3 = nn.GRUCell(args.h_dim, args.h_dim, bias=True)
        self.out1 = nn.Linear(args.h_dim, args.a_dim * args.n_obj)
        self.out2 = nn.Linear(args.h_dim, args.a_dim * args.n_obj)
        self.out3 = nn.Linear(args.h_dim, args.a_dim * args.n_obj)

    def forward(self, input):
        input = self.convert_type(input)
        h = self.Linear(input)
        h1 = self.GRU1(h)
        h2 = self.GRU2(h, h1)
        h3 = self.GRU3(h, h2)
        out1 = self.out1(h).view(-1, self.args.a_dim, self.args.n_obj)
        out2 = self.out2(h2).view(-1, self.args.a_dim, self.args.n_obj)
        out3 = self.out3(h3).view(-1, self.args.a_dim, self.args.n_obj)

        return [out1, out2, out3]

    def convert_type(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        if x.device != torch.device(self.args.device):
            x = x.to(self.args.device)
        return x

    def choose_action(self, o, w):
        o = self.convert_type(o)
        w = self.convert_type(w)
        input = torch.cat([o, w], dim=-1)
        print("input", input.shape)
        q = self.forward(input)
        print("q before T:", q[0])
        print("w:", w)
        # print("q[0]", q[0].shape, w.shape)
        # print("w:", w.view(-1, self.args.n_obj),"q:", q[0].view(-1, self.args.n_obj).T)
        for i in range(len(q)):
            # print(q[i])
            # print(w.view(-1, self.args.n_obj), q[i].view(-1, self.args.n_obj).T)
            q[i] = torch.cat([torch.matmul(w[t].view(-1, self.args.n_obj), q[i][t].view(-1, self.args.n_obj).T)
                              for t in range(self.args.n_threads)])
            # print("q[{0}]".format(i), q[i])
        # print("q:", q)
        # print(q[0].shape)
        actions = []
        for i in range(3):
            if random.random() > self.args.epsilon:
                actions.append(np.array([random.randint(0, self.args.a_dim - 1) for _ in range(self.args.n_threads)]))
            else:
                actions.append(torch.argmax(q[i], dim=-1).data.cpu().numpy())

        # print("actions:", actions)
        return actions

    def learn(self, sample):
        raise NotImplemented




def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", default=5)
    parser.add_argument("--gpu", default=False)
    parser.add_argument("--buffer_size", default=50000)
    parser.add_argument("--h_dim", default=128)
    parser.add_argument("--n_obj", default=2)
    parser.add_argument("--hyper_h1", default=64)
    parser.add_argument("--n_threads", default=1)
    parser.add_argument("--s_dim", default=5)
    parser.add_argument("--a_dim", default=10)

    return parser.parse_args()

if __name__ == '__main__':
    Q = Q_net(get_args())
    t = Q.parameters()
    print()


