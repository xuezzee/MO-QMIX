import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(params.s_dim, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, params.a_dim)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, params):
        self.eval_net, self.target_net = Net(params), Net(params)
        self.params = params
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, params.s_dim * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99, last_epoch=-1)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.params.a_dim)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.params.s_dim])
        b_a = torch.LongTensor(b_memory[:, self.params.s_dim:self.params.s_dim+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.params.s_dim+1:self.params.s_dim+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.params.s_dim:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

# dqn = DQN()
#
# print('\nCollecting experience...')
# for i_episode in range(400):
#     s = env.reset()
#     ep_r = 0
#     while True:
#         env.render()
#         a = dqn.choose_action(s)
#
#         # take action
#         s_, r, done, info = env.step(a)
#
#         # modify the reward
#         x, x_dot, theta, theta_dot = s_
#         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#         r = r1 + r2
#
#         dqn.store_transition(s, a, r, s_)
#
#         ep_r += r
#         if dqn.memory_counter > MEMORY_CAPACITY:
#             dqn.learn()
#             if done:
#                 print('Ep: ', i_episode,
#                       '| Ep_r: ', round(ep_r, 2))
#
#         if done:
#             break
#         s = s_

class DQNAgents():
    def __init__(self, params):
        self.params = params
        self.agents = [DQN(params) for i in range(params.n_agents)]
        # N_ACTIONS = params.a_dim
        # N_STATES = params.s_dim

    def choose_action(self, state):
        action = []
        for i in range(self.params.n_agents):
            agent = self.agents[i]
            act = agent.choose_action(torch.Tensor(state[i]))
            action.append(act)

        return action

    def learn(self):
        for agent in self.agents:
            agent.learn()

    def store_transitions(self, transitions):
        for i in range(self.params.n_agents):
            temp = transitions[i]
            self.agents[i].store_transition(temp['state'], temp['action'], temp['reward'], temp['next_state'])

    def save(self, ep, path="./model/DQN/"):
        print("//////////////////////////////////////////////////////////////\n","saving model")
        state = {}
        for i in range(self.params.n_agents):
            state["eval_net%d"%i] = self.agents[i].eval_net.state_dict()
            state["targ_net%d"%i] = self.agents[i].target_net.state_dict()
            state["optim%d"%i] = self.agents[i].optimizer.state_dict()
        state["epoch"] = ep
        torch.save(state, path+"model.pth")
        print("//////////////////////////////////////////////////////////////\n","finish saving")

    def load(self, path="./model/DQN/"):
        state = torch.load(path+"model.pth")
        for i in range(self.params.n_agents):
            self.agents[i].eval_net.load_state_dict(state["eval_net%d"%i])
            self.agents[i].target_net.load_state_dict(state["targ_net%d"%i])
            self.agents[i].optimizer.load_state_dict(state["optim%d"%i])
        return state["epoch"]
