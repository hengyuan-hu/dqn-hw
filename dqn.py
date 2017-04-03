"""Main DQN agent."""
import time
import copy
import torch
import torch.nn
from torch.autograd import Variable
import numpy as np
import utils
from policy import GreedyEpsilonPolicy
import core
from collections import Counter
import os


class DQNAgent(object):
    """Class implementing DQN.

    Parameters
    ----------
    q_network: torch.nn.Module
    memory: replay memory.
    gamma: discount factor.
    target_update_freq: frequency to sync target and online qs
    num_burn_in: fill the memory before training starts
    use_double_dqn: boolean
    """
    def __init__(self,
                 q_network,
                 memory,
                 gamma,
                 target_update_freq,
                 use_double_dqn):
        self.online_q_net = q_network
        self.target_q_net = copy.deepcopy(q_network)

        self.replay_memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn

    def burn_in(self, env, num_burn_in):
        policy = GreedyEpsilonPolicy(1) # uniform policy
        dummy_q_values = np.zeros(env.num_actions)
        i = 0
        while i < num_burn_in or not env.end:
            if env.end:
                state = env.reset()
            action = policy(dummy_q_values)
            next_state, reward = env.step(action)
            self.replay_memory.append(state, action, reward, next_state, env.end)
            state = next_state
            i += 1
        print '%d frames are burned into the memory.' % i

    def target_q_values(self, states):
        utils.assert_eq(type(states), torch.cuda.FloatTensor)
        q_vals = self.target_q_net(Variable(states, volatile=True))
        return q_vals.data

    def online_q_values(self, states):
        utils.assert_eq(type(states), torch.cuda.FloatTensor)
        q_vals = self.online_q_net(Variable(states, volatile=True))
        return q_vals.data

    def select_action(self, states, policy):
        """Select the action based on the current state and ONLINE Q Network.

        states: Tensor with size: [batch_size, num_frames, frame_size, frame_size]
                states SHOULD BE preoprocessed
        policy: policy takes Q-values and return actions
        returns:  selected action, 1-d array (batch_size,)
        """
        # TODO: not efficient, avoid compute q val if greedy
        q_vals = self.online_q_values(states)
        # utils.assert_eq(q_vals.size()[0], 1)
        # q_vals = q_vals.view(q_vals.size()[1])
        # print q_vals.size()
        action = policy(q_vals.cpu().numpy())
        return action

    def _update_q_net(self, batch_size, logger):
        samples = self.replay_memory.sample(batch_size)
        x, a, y = core.samples_to_minibatch(samples, self)
        loss = self.online_q_net.train_step(x, a, y)
        logger.append('loss', loss)

    def train(self, batch_env, policy, batch_size, num_iters,
              eval_args, logger, output_path):
        log_per_iters = 1000

        states_gpu = torch.cuda.FloatTensor(*batch_env.batch_shape)
        states = np.zeros(batch_env.batch_shape, dtype=np.float32)
        ends = np.ones(batch_env.num_envs)
        total_rewards = 0
        t = time.time()
        for i in xrange(num_iters):
            if ends.any():
                states, ends = batch_env.reset(states, ends)

            states_gpu.copy_(torch.from_numpy(states))
            actions = self.select_action(states_gpu, policy)
            next_states, rewards, ends = batch_env.step(actions)
            self.replay_memory.batch_append(
                states, actions, rewards, next_states, ends)
            states = next_states
            total_rewards += sum(rewards)
            self._update_q_net(batch_size, logger)

            if (i+1) % log_per_iters == 0:
                fps = log_per_iters / (time.time() - t)
                log_msg = ('Iter: %d, Total Reward: %d; Fps: %.2f'
                           % (i+1, total_rewards, fps))
                print logger.log(log_msg)
                total_rewards = 0
                t = time.time()

            if (i+1) % eval_args['eval_per_iter'] == 0:
                eval_log = self.eval(eval_args['eval_env'],
                                     eval_args['eval_policy'],
                                     eval_args['num_episodes'])
                print logger.log(eval_log)

            if (i+1) % self.target_update_freq == 0:
                self.target_q_net = copy.deepcopy(self.online_q_net)

            if (i+1) % (num_iters/4) == 0:
                model_path = os.path.join(
                    output_path, 'net_%d.pth' % ((i+1)/(num_iters/4)))
                torch.save(self.online_q_net.state_dict(), model_path)

    def eval(self, env, policy, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        state_gpu = torch.cuda.FloatTensor(
            1, env.num_frames, env.frame_size, env.frame_size)
        state = env.reset()
        actions = np.zeros(env.num_actions)

        total_rewards = np.zeros(num_episodes)
        eps_idx = 0
        log = ''
        while eps_idx < num_episodes:
            state_gpu.copy_(torch.from_numpy(state.reshape(state_gpu.size())))
            action = self.select_action(state_gpu, policy)
            actions[action] += 1
            state, _  = env.step(action)

            if env.end:
                total_rewards[eps_idx] = env.total_reward
                eps_log = ('>>>Eval: [%d/%d], rewards: %s\n' %
                           (eps_idx+1, num_episodes, total_rewards[eps_idx]))
                log += eps_log
                if eps_idx < num_episodes-1: # leave last reset to next run
                    state = env.reset()
                eps_idx += 1

        eps_log = '>>>Eval: avg total rewards: %s\n' % total_rewards.mean()
        log += eps_log
        log += '>>>Eval: actions dist: %s\n' % list(actions/actions.sum())
        return log


class PredDQNAgent(DQNAgent):
    def target_q_values(self, states):
        utils.assert_eq(type(states), torch.cuda.FloatTensor)
        q_vals, feat, _ = self.target_q_net(Variable(states, volatile=True), False)
        return q_vals.data, feat.data

    def online_q_values(self, states):
        utils.assert_eq(type(states), torch.cuda.FloatTensor)
        q_vals, _, _ = self.online_q_net(Variable(states, volatile=True), False)
        return q_vals.data

    def _update_q_net(self, batch_size, logger):
        samples = self.replay_memory.sample(batch_size)
        x, a, y, target_feat = core.samples_to_minibatch(samples, self, True)
        q_loss, pred_loss = self.online_q_net.train_step(x, a, y, target_feat)
        logger.append('q_loss', q_loss)
        logger.append('pred_loss', pred_loss)
