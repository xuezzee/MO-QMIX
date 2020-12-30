import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
import sys
from pathlib import Path

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim=128, device='cpu'):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.device = device
        self.build_network()

    def build_network(self):
        self.linear1 = nn.Linear(self.s_dim, self.h_dim)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2 = nn.Linear(self.h_dim, self.h_dim)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear3 = nn.Linear(self.h_dim, self.a_dim)
        self.linear3.weight.data.normal_(0, 0.1)

    def forward(self, input):
        x = self.convert_type(input)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.softmax(self.linear3(x), dim=-1)
        return x

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.device)
        return x


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim=128, device='cpu'):
        super(Critic, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.device = device
        self.build_network()

    def build_network(self):
        self.linear1 = nn.Linear(self.s_dim, self.h_dim)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2 = nn.Linear(self.h_dim, self.h_dim)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear3 = nn.Linear(self.h_dim, 1)
        self.linear3.weight.data.normal_(0, 0.1)

    def forward(self, input):
        x = self.convert_type(input)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.device)
        return x

class ActorCritic():
    def __init__(self, params):
        self.params = params
        self.device = params.device
        self.step_ent = 10
        self.actor = Actor(params.s_dim, params.a_dim, device=params.device).to(self.device)
        self.critic = Critic(params.s_dim, params.a_dim, device=params.device).to(self.device)
        self.optim = {'optA':torch.optim.Adam(self.actor.parameters(), lr=0.0001),
                      'optC':torch.optim.Adam(self.critic.parameters(), lr=0.001)}
        self.lr_scheduler = {'lrA':torch.optim.lr_scheduler.StepLR(self.optim['optA'], 1000, 0.99, last_epoch=-1),
                             'lrC':torch.optim.lr_scheduler.StepLR(self.optim['optC'], 1000, 0.99, last_epoch=-1)}

    def choose_action(self, state):
        self.prob = self.actor(state)
        # print("prob: {0}".format(self.prob))
        m = Categorical(self.prob)
        action = m.sample()
        return int(action.data.cpu().numpy())

    def cal_td(self, transition):
        s = torch.Tensor(transition['state']).to(self.device)
        s_ = torch.Tensor(transition['next_state']).to(self.device)
        r = torch.Tensor(transition['reward']).to(self.device)
        q = self.critic(s)
        q_ = self.critic(s_)
        td_err = q - r - self.params.gamma * q_.detach()
        return td_err

    def learnActor(self, transition, td_err):
        def entropy(dist):
            ent = 0
            for p in dist:
                ent -= p * torch.log(p)
            return ent

       # log_prob = torch.log(self.prob[0][transition['action']])
        log_prob = torch.log(self.prob[transition['action']])
        self.step_ent = self.step_ent * 0.99
        loss = - (log_prob * td_err) - .10 * entropy(self.prob)
        self.optim["optA"].zero_grad()
        loss.backward()
        self.optim["optA"].step()
        self.lr_scheduler["lrA"].step()

    def learnCritic(self, transition):
        td_err = self.cal_td(transition)
        # print("td_loss:", td_err)
        loss = td_err.pow(2)
        self.optim['optC'].zero_grad()
        loss.backward()
        self.optim['optC'].step()
        self.lr_scheduler['lrC'].step()
        return td_err

    def learn(self, transition):
        td_err = self.learnCritic(transition)
        # self.learnActor(transition, td_err.detach())
        self.learnActor(transition, transition['reward'][0])


class ACAgents():
    def __init__(self, params):
        self.params = params
        self.device = params
        self.agents = [ActorCritic(params) for i in range(params.n_agents)]

    def choose_action(self, state):
        actions = []
        for i in range(self.params.n_agents):
            agent = self.agents[i]
            actions.append(agent.choose_action(state[i]))

        return actions

    def learn(self, transition):
        for i in range(self.params.n_agents):
            agent = self.agents[i]
            agent.learn(transition=transition[i])

    def save(self, ep, path="./model/AC/"):
        print("model is saving")
        state = {}
        for i in range(self.params.n_agents):
            state["actor%d"%i] = self.agents[i].actor.state_dict()
            state["critic%d" % i] = self.agents[i].critic.state_dict()
            state["optA%d"%i] = self.agents[i].optim["optA"].state_dict()
            state["optC%d"%i] = self.agents[i].optim["optC"].state_dict()
        state["epoch"] = ep
        state["agent_num"] = self.params.n_agents
        torch.save(state, path+"model.pth")
        print("model finishes saving")

    def load(self, path="./model/AC/", map_location='cpu'):
        state = torch.load(path+"model.pth", map_location=map_location)
        for i in range(self.params.n_agents):
            self.agents[i].actor.load_state_dict(state["actor%d"%i])
            self.agents[i].critic.load_state_dict(state["critic%d"%i])
            self.agents[i].optim["optA"].load_state_dict(state["optA%d"%i])
            self.agents[i].optim["optC"].load_state_dict(state["optC%d"%i])
        return state["epoch"]



