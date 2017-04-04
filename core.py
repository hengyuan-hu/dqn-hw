"""Core classes."""
import utils
import random
import torch
import time
import numpy as np


class BatchReplayMemory(object):
    def __init__(self, max_size, state_shape):
        self.max_size = max_size
        self.states = np.zeros((max_size,) + state_shape, dtype=np.uint8)
        self.next_states = np.zeros((max_size,) + state_shape, dtype=np.uint8)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int64)
        self.ends = np.zeros(max_size, dtype=np.float32)

        self.next_idx = 0
        self.num_samples = 0

    def __len__(self):
        return len(self.next_idx)

    def _get_next_batch_idxs(self, batch_size):
        idxs = np.array(range(self.next_idx, self.next_idx+batch_size))
        idxs %= self.max_size
        self.next_idx = idxs[-1] + 1
        return idxs

    def append(self, state, action, reward, next_state, end):
        self.batch_append(
            np.array([state]), np.array([action]), np.array([reward]),
            np.array([next_state]), np.array([end]))

    def batch_append(self, states, actions, rewards, next_states, ends):
        idxs = self._get_next_batch_idxs(len(states))
        self.states[idxs] = states.astype(np.uint8)
        self.next_states[idxs] = next_states.astype(np.uint8)
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.ends[idxs] = ends
        self.num_samples = min(self.num_samples+len(states), self.max_size)

    def sample(self, batch_size, indexes=None):
        """Simpliest uniform sampling (w/o replacement) to produce a batch."""
        assert batch_size < self.num_samples, 'no enough samples to sample from'
        assert indexes is None, 'not supported yet'
        idxs = random.sample(xrange(self.num_samples), batch_size)
        # idxs = range(self.num_samples)[:batch_size]
        samples = {'states': self.states[idxs].astype(np.float32),
                   'next_states': self.next_states[idxs].astype(np.float32),
                   'actions': self.actions[idxs],
                   'rewards': self.rewards[idxs],
                   'ends': self.ends[idxs]}
        return samples


def _samples_to_tensors(samples):
    states = torch.from_numpy(samples['states']).cuda()
    next_states = torch.from_numpy(samples['next_states']).cuda()
    actions = torch.from_numpy(samples['actions']).view(-1, 1).cuda()
    rewards = torch.from_numpy(samples['rewards']).view(-1, 1).cuda()
    non_ends = 1 - torch.from_numpy(samples['ends']).view(-1, 1).cuda()
    return states, actions, rewards, next_states, non_ends


# def test_samples():
#     state_shape = (3, 5, 5)

#     def gen_samples(batch_size):
#         states = np.random.uniform(size=(batch_size,) + state_shape) * 255
#         next_states = np.random.uniform(size=(batch_size,) + state_shape) * 255
#         rewards = np.random.uniform(size=batch_size)
#         actions = np.random.uniform(size=batch_size)
#         ends = np.random.uniform(size=batch_size).round()
#         return states, next_states, rewards, actions, ends

#     mem = ReplayMemory(10)
#     batch_mem = BatchReplayMemory(10, state_shape)

#     for i in range(10):
#         states, next_states, rewards, actions, ends = gen_samples(4)
#         # print states[0].astype(np.uint8)
#         mem.batch_append(states, actions, rewards, next_states, ends)
#         batch_mem.batch_append(states, actions, rewards, next_states, ends)
#         # TODO: /255.0 issue, test two samples
#         s1 = mem.sample(2)
#         s2 = batch_mem.sample(2)
#         states, actions, rewards, next_states, non_ends = _samples_to_tensors(s1)
#         states2, actions2, rewards2, next_states2, non_ends2 = _samples_to_tensors2(s2)

#         assert (states.cpu().numpy() == states2.cpu().numpy()).all()
#         assert (next_states.cpu().numpy()
#                 == next_states2.cpu().numpy()).all()
#         assert (actions.cpu().numpy() == actions2.cpu().numpy()).all()
#         assert (rewards.cpu().numpy() == rewards2.cpu().numpy()).all()
#         assert (non_ends.cpu().numpy() == non_ends2.cpu().numpy()).all()

#     print 'pass'


def samples_to_minibatch(samples, q_agent, need_target_feat=False):
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
    states, actions, ys, next_states, non_ends = _samples_to_tensors(samples)

    if need_target_feat:
        target_q_vals, target_feat = q_agent.target_q_values(next_states)
    else:
        target_q_vals = q_agent.target_q_values(next_states)
    n_actions = target_q_vals.size(1)
    actions_one_hot = torch.zeros(len(states), n_actions).cuda()
    actions_one_hot.scatter_(1, actions, 1)

    if q_agent.use_double_dqn:
        online_q_values = q_agent.online_q_values(next_states)
        next_actions = online_q_values.max(1)[1] # argmax
        next_actions_one_hot = torch.zeros(len(states), n_actions).cuda()
        next_actions_one_hot.scatter_(1, next_actions, 1)
        next_qs = (target_q_vals * next_actions_one_hot).sum(1)
    else:
        next_qs = target_q_vals.max(1)[0] # max returns a pair
    ys.add_(next_qs.mul_(non_ends).mul_(q_agent.gamma))

    if need_target_feat:
        # TODO: for terminal state: target_feat.mul_(non_ends) ???
        return states, actions_one_hot, ys, target_feat
    else:
        return states, actions_one_hot, ys
