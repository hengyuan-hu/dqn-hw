"""Wrapper of OpenAI Gym enviroment"""
import gym
import cv2


def preprocess_frame(observ, output_size):
    # to grayscale, resize, scale to [0, 1]
    # !!! NO CROP (different from original)
    gray = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, (output_size, output_size))
    output = output.astype(np.float32) / 255.0
    return output


class Environment(object):
    def __init__(self, name, frame_size):
        self.env = gym.make(name)
        self.frame_size = frame_size

        self.end = True

    def reset(self):
        self.env.reset()
        self.end = False

    @property
    def num_actions(self):
        return self.env.action_space.n

    def step(self, action):
        assert not self.end
        observ, reward, self.end, info = env.step(action)


if __name__ == '__main__':
     env = gym.make('SpaceInvaders-v0')
