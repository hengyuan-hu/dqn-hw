"""Core classes."""
import utils
import random
import torch

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
    batch_term, batch = [], []
    for sample in samples:
        batch_term.append(sample) if sample.end else batch.append(sample)

    if batch:
        batch = [ (s.state.tolist(), [s.action], [s.reward]) for s in batch]
        xs, actions, ys = zip(*batch) # (32L, 4L, 84L, 84L) (32L, 1L) (32L, 1L)
        xs = torch.cuda.FloatTensor(xs)
        actions = torch.cuda.LongTensor(actions)
        ys = torch.cuda.FloatTensor(ys)

        q_values = q_agent.target_q_values(xs) # Tensor (b, n_actions)

        n_actions = q_values.size()[1]
        max_qs = q_values.max(1)[0] # FloatTensor
        ys += max_qs.mul(q_agent.gamma)

    if batch_term:
        batch_term = [ (s.state.tolist(), [s.action], [s.reward]) for s in batch_term]
        xs_term, actions_term, ys_term = zip(*batch_term)
        xs_term = torch.cuda.FloatTensor(xs_term)
        actions_term = torch.cuda.LongTensor(actions_term)
        ys_term = torch.cuda.FloatTensor(ys_term)
        if batch:
            xs = torch.cat((xs, xs_term))
            actions = torch.cat((actions, actions_term))
            ys = torch.cat((ys, ys_term))
        else:
            xs = xs_term
            actions = actions_term
            ys = ys_term
    # now x, a, y must contain all samples
    # convert to one-hot
    actions = torch.zeros(len(samples), n_actions).scatter_(1, actions, 1)

    assert xs.size(0)==len(samples)
    return xs, actions, ys

# Preprocess is done in the Env class
# class Preprocessor:
#     """Preprocessor base class.

#     This is a suggested interface for the preprocessing steps. You may
#     implement any of these functions. Feel free to add or change the
#     interface to suit your needs.

#     Preprocessor can be used to perform some fixed operations on the
#     raw state from an environment. For example, in ConvNet based
#     networks which use image as the raw state, it is often useful to
#     convert the image to greyscale or downsample the image.

#     Preprocessors are implemented as class so that they can have
#     internal state. This can be useful for things like the
#     AtariPreproccessor which maxes over k frames.

#     If you're using internal states, such as for keeping a sequence of
#     inputs like in Atari, you should probably call reset when a new
#     episode begins so that state doesn't leak in from episode to
#     episode.
#     """

#     def process_state_for_network(self, state):
#         """Preprocess the given state before giving it to the network.

#         Should be called just before the action is selected.

#         This is a different method from the process_state_for_memory
#         because the replay memory may require a different storage
#         format to reduce memory usage. For example, storing images as
#         uint8 in memory is a lot more efficient thant float32, but the
#         networks work better with floating point images.

#         Parameters
#         ----------
#         state: np.ndarray
#           Generally a numpy array. A single state from an environment.

#         Returns
#         -------
#         processed_state: np.ndarray
#           Generally a numpy array. The state after processing. Can be
#           modified in anyway.

#         """
#         return state

#     def process_state_for_memory(self, state):
#         """Preprocess the given state before giving it to the replay memory.

#         Should be called just before appending this to the replay memory.

#         This is a different method from the process_state_for_network
#         because the replay memory may require a different storage
#         format to reduce memory usage. For example, storing images as
#         uint8 in memory and the network expecting images in floating
#         point.

#         Parameters
#         ----------
#         state: np.ndarray
#           A single state from an environmnet. Generally a numpy array.

#         Returns
#         -------
#         processed_state: np.ndarray
#           Generally a numpy array. The state after processing. Can be
#           modified in any manner.

#         """
#         return state

#     def process_batch(self, samples):
#         """Process batch of samples.

#         If your replay memory storage format is different than your
#         network input, you may want to apply this function to your
#         sampled batch before running it through your update function.

#         Parameters
#         ----------
#         samples: list(tensorflow_rl.core.Sample)
#           List of samples to process

#         Returns
#         -------
#         processed_samples: list(tensorflow_rl.core.Sample)
#           Samples after processing. Can be modified in anyways, but
#           the list length will generally stay the same.
#         """
#         return samples

#     def process_reward(self, reward):
#         """Process the reward.

#         Useful for things like reward clipping. The Atari environments
#         from DQN paper do this. Instead of taking real score, they
#         take the sign of the delta of the score.

#         Parameters
#         ----------
#         reward: float
#           Reward to process

#         Returns
#         -------
#         processed_reward: float
#           The processed reward
#         """
#         return reward

#     def reset(self):
#         """Reset any internal state.

#         Will be called at the start of every new episode. Makes it
#         possible to do history snapshots.
#         """
#         pass
