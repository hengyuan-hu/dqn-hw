"""Wrapper of OpenAI Gym enviroment"""
import os
from collections import deque
import cv2
import numpy as np
from ale_python_interface import ALEInterface
# import time


glb_counter = 0


def preprocess_frame(observ, output_size):
    gray = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, (output_size, output_size))
    output = output.astype(np.float32, copy=False)
    return output


class Environment(object):
    def __init__(self,
                 rom_file,
                 frame_skip,
                 num_frames,
                 frame_size,
                 no_op_start,
                 rand_seed,
                 dead_as_eoe):
        self.ale = self._init_ale(rand_seed, rom_file)
        # normally (160, 210)
        self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.actions = self.ale.getMinimalActionSet()

        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.no_op_start = no_op_start
        self.dead_as_eoe = dead_as_eoe

        self.clipped_reward = 0
        self.total_reward = 0
        self.prev_screen = np.zeros(
            (self.screen_height, self.screen_width, 3), dtype=np.float32)
        self.frame_queue = deque(maxlen=num_frames)
        self.end = True

    @staticmethod
    def _init_ale(rand_seed, rom_file):
        assert os.path.exists(rom_file), '%s does not exists.'
        ale = ALEInterface()
        ale.setInt('random_seed', rand_seed)
        ale.setBool('showinfo', False)
        ale.setInt('frame_skip', 1)
        ale.setFloat('repeat_action_probability', 0.0)
        ale.setBool('color_averaging', False)
        ale.loadROM(rom_file)
        return ale

    @property
    def num_actions(self):
        return len(self.actions)

    def _get_current_frame(self):
        # global glb_counter
        screen = self.ale.getScreenRGB()
        max_screen = np.maximum(self.prev_screen, screen)
        frame = preprocess_frame(max_screen, self.frame_size)
        # cv2.imwrite('test_env/%d.png' % glb_counter, cv2.resize(frame, (800, 800)))
        # glb_counter += 1
        return frame

    def reset(self):
        self.end = False
        self.clipped_reward = 0
        self.total_reward = 0

        for _ in range(self.num_frames - 1):
            self.frame_queue.append(
                np.zeros((self.frame_size, self.frame_size), dtype=np.float32))

        if self.ale.game_over():
            self.ale.reset_game()
        else:
            assert self.dead_as_eoe
        n = np.random.randint(0, self.no_op_start)
        for i in range(n):
            if i == n - 1:
                self.prev_screen = self.ale.getScreenRGB()
            self.ale.act(0)

        self.frame_queue.append(self._get_current_frame())

        return np.array(self.frame_queue)

    def step(self, action_idx):
        """Perform action and return frame sequence and reward.
        Return:
        state: [frames] of length num_frames, 0 if fewer is available
        reward: float
        """
        assert not self.end
        reward = 0
        clipped_reward = 0
        old_lives = self.ale.lives()
        for _ in range(self.frame_skip):
            self.prev_screen = self.ale.getScreenRGB()
            r = self.ale.act(self.actions[action_idx])
            reward += r
            clipped_reward += np.sign(r)
            dead = (self.ale.lives() < old_lives)
            if self.ale.game_over() or (self.dead_as_eoe and dead):
                self.end = True
                break

        self.frame_queue.append(self._get_current_frame())
        self.total_reward += reward
        self.clipped_reward += clipped_reward
        return np.array(self.frame_queue), clipped_reward


if __name__ == '__main__':
    env = Environment('roms/space_invaders.bin', 4, 4, 84, 30, 33)

    state = env.reset()
    i = 0
    while not env.end:
        action = np.random.randint(0, env.num_actions)
        state, reward = env.step(action)
        # print state.shape
        # print state.max(), state.min()
        i += 1
    print 'total_reward:', env.total_reward
    print 'clipped_reward', env.clipped_reward
    print 'total steps:', i

    state = env.reset()
    i = 0
    while not env.end:
        action = np.random.randint(0, env.num_actions)
        state, reward = env.step(action)
        if i == 100:
            for idx, f in enumerate(state):
                cv2.imwrite('test_env/f%d.png' % idx,
                            cv2.resize(f, (800, 800)))
        i += 1
    print 'total_reward:', env.total_reward
    print 'clipped_reward', env.clipped_reward
    print 'total steps:', i
