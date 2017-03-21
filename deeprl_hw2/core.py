"""Core classes."""
import utils
import random
import torch
import time
import numpy as np

class Sample(object):
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, state, action, reward, next_state, end):
        utils.assert_eq(type(state), type(next_state))
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.end = end

    def __str__(self):
        info = ('S(mean): %3.4f, A: %s, R: %s, NS(mean): %3.4f, End: %s'
                % (self.state.mean(), self.action, self.reward,
                   self.next_state.mean(), self.end))
        return info

    def __repr__(self):
        """this is bad"""
        return self.__str__()


class ReplayMemory(object):
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size):
        """Setup memory.

        You should specify the maximum size of the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
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

    def end_episode(self, final_state, is_terminal):
        """TODO: what is this?"""
        raise NotImplementedError('This method should be overridden')

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

    return: Tensors that can be directly used by q_network
        xs: (b, ?) FloatTensor
        as: (b, n_actions) one-hot FloatTensor
        ys: (b, 1) FloatTensor
    """
    assert len(samples)>0
    dummy = samples[0]

    xs = np.zeros((len(samples),)+ dummy.state.shape, dtype=np.float32)
    next_states = np.zeros_like(xs)
    ys = np.zeros((len(samples), 1), dtype=np.float32)
    actions = np.zeros_like(ys, dtype=np.int64)
    ends = np.zeros_like(ys)
    for i, s in enumerate(samples):
        xs[i] = s.state
        next_states[i] = s.next_state
        ys[i] = s.reward
        actions[i] = s.action
        ends[i] = 0.0 if s.end else 1.0

    xs = torch.from_numpy(xs).cuda()
    next_states = torch.from_numpy(next_states).cuda()
    actions = torch.from_numpy(actions).cuda()
    ys = torch.from_numpy(ys).cuda()
    ends = torch.from_numpy(ends).cuda()

    q_values = q_agent.target_q_values(next_states) # Tensor (b, n_actions)
    n_actions = q_values.size(1)
    max_qs = q_values.max(1)[0] # max returns a pair
    ys += max_qs.mul(q_agent.gamma).mul(ends)

    # convert to one-hot
    actions = torch.zeros(len(samples), n_actions).cuda().scatter_(1, actions, 1)

    assert xs.size(0)==len(samples)
    return xs, actions, ys

