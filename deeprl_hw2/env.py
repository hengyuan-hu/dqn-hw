"""Wrapper of OpenAI Gym enviroment"""
import gym
from gym import wrappers
import cv2
from collections import deque
import numpy as np

def preprocess_frame(observ, output_size):
    # to grayscale, resize, scale to [0, 1]
    # !!! NO CROP (different from original)
    gray = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, (output_size, output_size))
    output = output.astype(np.float32) / 255.0
    return output


class Environment(object):
    def __init__(self, name, num_frames, frame_size, mnt_path='./monitor'):
        assert num_frames>0
        self.env = gym.make(name)
        # force=True for debug easiness, need care before actual training
        self.env = wrappers.Monitor(self.env, mnt_path, force=True)
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.end = True
        self.total_reward = 0.0
        self.frames_queue = deque(maxlen=4)
        self._reset()

    def _reset(self):
        """reset env and frame queue"""
        initial_queue = [np.zeros((self.frame_size, self.frame_size))
                         for _ in range(self.num_frames-1)]
        self.frames_queue.extend(initial_queue)
        initial_state = preprocess_frame(self.env.reset(), self.frame_size)
        self.frames_queue.append(initial_state)

    def reset(self):
        self.end = False
        self.total_reward = 0.0
        self._reset()

        # return initial state
        return np.array(self.frames_queue)

    def render(self):
        self.env.render()

    @property
    def num_actions(self):
        return self.env.action_space.n

    def step(self, action):
        """Perform action and return frame sequence and reward.
        Return:
        state: [frames] of length num_frames, 0 if fewer is available
        reward: float
        """
        assert not self.end
        obs, reward, self.end, info = self.env.step(action)
        obs = preprocess_frame(obs, self.frame_size)

        self.frames_queue.append(obs) # left is automatically popped
        self.total_reward += reward

        # clip reward
        if reward != 0.0:
            reward = 1.0 if reward > 0 else -1.0

        return np.array(self.frames_queue), reward

if __name__ == '__main__':
     env = gym.make('SpaceInvaders-v0')
