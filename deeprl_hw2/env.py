"""Wrapper of OpenAI Gym enviroment"""
import gym
from gym import wrappers
import cv2
from collections import deque
import numpy as np
from scipy.misc import imsave

def preprocess_frame(observ, output_size):
    # to grayscale, resize, scale to [0, 1]
    # !!! NO CROP (different from original)
    gray = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, (output_size, output_size))
    output = output.astype(np.float32)
    output[output>0] = 1.0
    imsave('frame.png', output)
    return output

SPACE_INVADERS_INIT_LIVES = 3

class Environment(object):
    def __init__(self, name, num_frames, frame_size, record=False, mnt_path=None,
                 video_callable=None, write_upon_reset=True, negative_dead_reward=False):
        assert num_frames>0
        self.env = gym.make(name)
        self.name = name
        if record:
            def capture_all(episode_id):
                return True
            def capture_every_twenty(episode_id):
                return (episode_id+1) % 20==0
            if mnt_path is None:
                mnt_path = './monitor/'
            if video_callable is None:
                video_callable = capture_every_twenty
            # force=True for debug easiness, need care before actual training
            self.env = wrappers.Monitor(self.env, mnt_path, force=True,
                                        video_callable=video_callable,
                                        write_upon_reset=write_upon_reset)
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.end = True
        self.total_reward = 0.0
        self.frames_queue = deque(maxlen=4)
        self.negative_dead_reward = negative_dead_reward
        # leave reset to user, as in env

    def reset(self):
        """reset env and frame queue, return initial state """
        self.end = False
        self.total_reward = 0.0
        initial_queue = [np.zeros((self.frame_size, self.frame_size))
                         for _ in range(self.num_frames-1)]
        self.frames_queue.extend(initial_queue)
        initial_state = self.env.reset()
        initial_state = preprocess_frame(initial_state, self.frame_size)
        self.frames_queue.append(initial_state)

        if self.negative_dead_reward and self.name == 'SpaceInvaders-v0':
            self.lives = SPACE_INVADERS_INIT_LIVES

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
        reward = np.sign(reward)

        # hack reward to be negative if life lost
        if self.negative_dead_reward and self.name=='SpaceInvaders-v0' and info['ale.lives']<self.lives:
            reward = -1.0
            self.lives = info['ale.lives']

        return np.array(self.frames_queue), reward#, self.end, info

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

if __name__ == '__main__':
     env = gym.make('SpaceInvaders-v0')
     """{
         0: 'Noop',
         1: 'Fire',
         2: 'Right',
         3: 'Left',
         4: 'RightFire',
         5: 'LeftFire'
     }"""
