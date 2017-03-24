"""Core classes."""
import utils
import random
import torch
import time
import numpy as np


class Sample(object):
    def __init__(self, state, action, reward, next_state, end):
        utils.assert_eq(type(state), type(next_state))
        # merge two states internally and clip to uint8 to save memory
        self._packed_frames = np.packbits(
            np.vstack([state, next_state]).astype(np.uint8, copy=False),axis=0)
        self.action = action
        self.reward = reward
        self.end = end

    def get_state_and_next_state(self):
        unpacked_frames = np.unpackbits(self._packed_frames, axis=0)
        unpacked_frames = unpacked_frames.astype(np.float32, copy=False)
        state = unpacked_frames[:4]
        next_state = unpacked_frames[4:]
        return state, next_state

    def __repr__(self):
        state, next_state = self.get_state_and_next_state()
        info = ('S(mean): %3.4f, A: %s, R: %s, NS(mean): %3.4f, End: %s'
                % (state.mean(), self.action, self.reward,
                   next_state.mean(), self.end))
        return info


class ReplayMemory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.samples = []
        self.oldest_idx = 0

    def __len__(self):
        return len(self.samples)

    def _evict(self):
        """Simplest FIFO eviction scheme."""
        to_evict = self.oldest_idx
        self.oldest_idx = (self.oldest_idx + 1) % self.max_size
        return to_evict

    def append(self, state, action, reward, next_state, end):
        assert len(self.samples) <= self.max_size
        new_sample = Sample(state, action, reward, next_state, end)
        if len(self.samples) == self.max_size:
            avail_slot = self._evict()
            self.samples[avail_slot] = new_sample
        else:
            self.samples.append(new_sample)

    def sample(self, batch_size, indexes=None):
        """Simpliest uniform sampling (w/o replacement) to produce a batch."""
        assert batch_size < len(self.samples), 'no enough samples to sample from'
        assert indexes is None, 'not supported yet'
        return random.sample(self.samples, batch_size)

    def clear(self):
        self.samples = []
        self.oldest_idx = 0


def samples_to_minibatch(samples, q_agent):
    """[samples] -> minibatch (xs, as, ys)
    convert [sample.state] to input tensor xs
    compute target tensor ys according to whether terminate and q_network
    it is possible to have only one kind of sample (all term/non-term)
    q_agent.use_double_dqn:
        True: y = r + targetQ(s', argmax_a'(onlineQ(s',a')))
        False: y = r + max_a'(onlineQ(s',a'))
    return: Tensors that can be directly used by q_network
        xs: (b, ?) FloatTensor
        as: (b, n_actions) one-hot FloatTensor
        ys: (b, 1) FloatTensor
    """
    assert len(samples) > 0
    dummy_state, _ = samples[0].get_state_and_next_state()

    states = np.zeros((len(samples),) + dummy_state.shape, dtype=np.float32)
    next_states = np.zeros_like(states)
    ys = np.zeros((len(samples), 1), dtype=np.float32)
    actions = np.zeros_like(ys, dtype=np.int64)
    non_ends = np.zeros_like(ys)
    for i, s in enumerate(samples):
        states[i], next_states[i] = s.get_state_and_next_state()
        ys[i] = s.reward
        actions[i] = s.action
        non_ends[i] = 0.0 if s.end else 1.0

    xs = torch.from_numpy(states).cuda()
    next_states = torch.from_numpy(next_states).cuda()
    actions = torch.from_numpy(actions).cuda()
    ys = torch.from_numpy(ys).cuda()
    non_ends = torch.from_numpy(non_ends).cuda()

    target_q_vals, next_feat, _ = q_agent.target_q_forward(next_states)
    n_actions = target_q_vals.size(1)
    actions_mask = torch.zeros(len(samples), n_actions).cuda()
    if q_agent.use_double_dqn:
        assert False, 'not checked yet'
        online_q_values = q_agent.online_q_values(next_states)
        next_actions = online_q_values.max(1)[1] # argmax
        next_actions = actions_mask.scatter_(1, next_actions, 1) # one-hot
        next_qs = target_q_vals.mul_(next_actions).sum(1)
    else:
        next_qs = target_q_vals.max(1)[0] # max returns a pair
    ys.add_(next_qs.mul_(non_ends).mul_(q_agent.gamma))
    # convert to one-hot
    actions_mask = torch.zeros(len(samples), n_actions).cuda() # reset to 0
    actions = actions_mask.scatter_(1, actions, 1) # scatter only set 1s

    assert xs.size(0) == len(samples)
    return xs, actions, ys, next_feat
