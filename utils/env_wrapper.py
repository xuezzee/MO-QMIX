import torch
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_func, args):
    parent_remote.close()
    env = env_func(args)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))

class EnvWrapper():
    def __init__(self, args, env_func):
        self.args = args
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.threads)])
        self.processes = [Process(target=worker, args=(work_remote, remote, env_func, args))
            for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.processes:
            p.daemon = True
            p.start()

        self.remotes[0].send(('get_space', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        tup = zip(*[self.remotes[i].recv() for i in range(self.args.threads)])

        return tup

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        tup = zip(*[self.remotes[i].recv() for i in range(self.args.threads)])

        return tup
