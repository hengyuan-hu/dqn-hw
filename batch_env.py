import numpy as np
from env import Environment
import utils


class BatchEnvironment(object):
    def __init__(self, num_envs, name, num_frames, frame_size, neg_dead_reward):
        self.envs = [Environment(name, num_frames, frame_size,
                                 negative_dead_reward=neg_dead_reward)
                     for _ in range(num_envs)]
        self.batch_shape = (num_envs, num_frames, frame_size, frame_size)
        self.num_envs = num_envs
        self.num_actions = self.envs[0].num_actions

    def __getitem__(self, idx):
        return self.envs[idx]

    def __len__(self):
        return self.num_envs

    def reset(self, states, ends):
        for i, env in enumerate(self.envs):
            if env.end:
                utils.assert_eq(ends[i], 1.0)
                states[i] = env.reset()
                ends[i] = np.float32(env.end)
        return states, ends

    def step(self, actions):
        states = np.zeros(self.batch_shape, dtype=np.float32)
        ends = np.zeros(self.num_envs, dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        for i, env in enumerate(self.envs):
            assert not env.end
            states[i], rewards[i] = env.step(actions[i])
            ends[i] = np.float32(env.end)
        return states, rewards, ends

    def close(self):
        for env in self.envs:
            env.close()
