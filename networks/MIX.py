import torch
from torch import nn
import torch.nn.functional as F

class HyperNet(nn.Module):
    def __init__(self, args):
        super(HyperNet, self).__init__()
        self.args = args
        self.hyper_w1 = nn.Sequential(
            nn.Linear(args.s_dim + args.n_obj, 128),
            nn.ReLU(),
            nn.Linear(128, args.n_agents * args.hyper_h1),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(args.s_dim + args.n_obj, 128),
            nn.ReLU(),
            nn.Linear(128, args.hyper_h1),
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(args.s_dim + args.n_obj, 128),
            nn.ReLU(),
            nn.Linear(128, args.hyper_h1)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(args.s_dim + args.n_obj, 128),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, w):
        s = self.convert_type(s)
        w = self.convert_type(w)
        x = torch.cat([s, w])
        w1 = self.hyper_w1(x).view(-1, self.args.hyper_h1)
        w2 = self.hyper_w2(x).view(-1, self.args.hyper_h2)
        b1 = self.hyper_b1(x)
        b2 = self.hyper_b2(x)

        return w1, w2, b1, b2

    def Q_tot(self, s, w, Q):
        input = []
        for q in Q:
            input.append(torch.bmm(w, q))
        input = torch.cat(input)
        w1, w2, b1, b2 = self.forward(s, w)
        h1 = F.relu(torch.bmm(w1, input) + b1)
        out = F.relu(torch.bmm(w2, h1) + b2)

        return torch.bmm(w, out)

    def convert_type(self, input):
        if not isinstance(input, torch.Tensor):
            input = torch.Tensor(input)
        if input.device() != torch.device(self.args.device):
            input = input.to(self.device)

        return input

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
    Q = HyperNet(get_args())
    t = Q.parameters()
    print()



