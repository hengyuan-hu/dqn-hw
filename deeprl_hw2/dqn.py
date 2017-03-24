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

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    use_double_dqn: boolean
      Whether to use target q or online q to calculate next_state action
      during sampling.
    use_double_q: boolean
      Whether to occasionally flip target and online. Only for Linear
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 use_double_dqn):
        self.online_q_net = q_network
        self.target_q_net = copy.deepcopy(q_network)

        self.replay_memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.use_double_dqn = use_double_dqn

    def _burn_in(self, env):
        policy = GreedyEpsilonPolicy(1) # uniform policy
        dummy_q_values = np.zeros(env.num_actions)
        for i in xrange(self.num_burn_in):
            if env.end:
                state = env.reset()
            action = policy(dummy_q_values)
            next_state, reward = env.step(action)
            self.replay_memory.append(state, action, reward, next_state, env.end)
            state = next_state
        return state

    def target_q_values(self, states):
        """Given a batch of states calculate the Q-values.

        states: Tensor with size: [batch_size, num_frames, frame_size, frame_size]
        return: Tensor with Q values, evaluated with target_q_net
        """
        utils.assert_eq(type(states), torch.cuda.FloatTensor)
        q_vals = self.target_q_net(Variable(states, volatile=True)).data
        utils.assert_eq(type(q_vals), torch.cuda.FloatTensor)
        return q_vals

    def online_q_values(self, states):
        utils.assert_eq(type(states), torch.cuda.FloatTensor)
        q_vals = self.online_q_net(Variable(states, volatile=True)).data
        utils.assert_eq(type(q_vals), torch.cuda.FloatTensor)
        return q_vals

    def select_action(self, states, policy):
        """Select the action based on the current state and ONLINE Q Network.

        states: Tensor with size: [batch_size, num_frames, frame_size, frame_size]
                states SHOULD BE preoprocessed
        policy: policy takes Q-values and return actions
        returns:  selected action, 1-d array (batch_size,)
        """
        # TODO: not efficient, avoid compute q val if greedy
        q_vals = self.online_q_values(states)
        utils.assert_eq(q_vals.size()[0], 1)
        q_vals = q_vals.view(q_vals.size()[1])
        action = policy(q_vals.cpu().numpy())
        return action

    def _update_q_net(self, batch_size):
        samples = self.replay_memory.sample(batch_size)
        x, a, y = core.samples_to_minibatch(samples, self)
        loss = self.online_q_net.train_step(x, a, y)
        return loss

    def train(self, env, policy, batch_size, num_iters, eval_args, output_path):
        # , max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        log_file = open(os.path.join(output_path, 'train_log.txt'), 'w')
        state_gpu = torch.cuda.FloatTensor(
            1, env.num_frames, env.frame_size, env.frame_size)
        state = self._burn_in(env)

        last_eval_milestone = 0
        num_episodes = 0
        rewards = []
        losses = []

        t = time.time()
        for i in xrange(num_iters):
            if env.end:
                # log and eval
                num_episodes += 1
                log = ('Episode: %d, Iter: %d, Reward Sum: %s; Loss: %s\n'
                       % (num_episodes, i+1, sum(rewards), np.mean(losses)))
                log += '\tTime taken: %s' % (time.time() - t)
                print '---memory size: ', len(self.replay_memory)
                print '---policy eps: ', policy.epsilon
                milestone = (i+1) / eval_args['eval_per_iter']
                if milestone > last_eval_milestone:
                    last_eval_milestone = milestone
                    log += '\n'
                    log += self.eval(
                        eval_args['eval_env'], eval_args['eval_policy'],
                        eval_args['num_episodes'])

                print log
                log_file.write(log+'\n')
                log_file.flush()

                # main task ...
                t = time.time()
                state = env.reset()
                rewards = []

            state_gpu.copy_(torch.from_numpy(state.reshape(state_gpu.size())))
            action = self.select_action(state_gpu, policy)
            next_state, reward = env.step(action)
            self.replay_memory.append(state, action, reward, next_state, env.end)
            state = next_state
            losses.append(self._update_q_net(batch_size))
            rewards.append(reward)

            if (i+1) % self.target_update_freq == 0:
                self.target_q_net = copy.deepcopy(self.online_q_net)
            if (i+1) % (num_iters/4) == 0:
                model_path = os.path.join(output_path, 'net_%d.pth' % ((i+1)/(num_iters/4)))
                torch.save(self.online_q_net.state_dict(), model_path)

        torch.save(self.online_q_net.state_dict(), os.path.join(output_path, 'net_final.pth'))
        log = self.eval(eval_args['eval_env'], eval_args['eval_policy'], eval_args['num_episodes_at_end'])
        eval_args['eval_env'].reset() # finish the recording for the very last episode
        log_file.write(log+'\n')
        log_file.flush()

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


class LinearQNAgent(DQNAgent):
    def __init__(self,
                 q_network,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 use_double_q):
        super(LinearQNAgent, self).__init__(
            q_network, memory, gamma, target_update_freq, num_burn_in, False)

        self.use_double_q = use_double_q
        if self.replay_memory is None:
            assert not self.num_burn_in, self.num_burn_in

    def train(self, env, policy, batch_size, num_iters, eval_args, output_path):
        log_file = open(os.path.join(output_path, 'train_log.txt'), 'w')
        state_gpu = torch.cuda.FloatTensor(
            1, env.num_frames, env.frame_size, env.frame_size)
        if self.replay_memory is not None:
            state = self._burn_in(env)

        num_episodes = 0
        last_eval_milestone = 0
        rewards = []
        losses = []

        t = time.time()
        for i in xrange(num_iters):
            if env.end:
                # log and eval
                num_episodes += 1
                log = ('Episode: %d, Iter: %d, Reward Sum: %s; Loss: %s\n'
                       % (num_episodes, i+1, sum(rewards), np.mean(losses)))
                log += '\tTime taken: %s' % (time.time() - t)
                if self.replay_memory:
                    print '---memory size: ', len(self.replay_memory)
                print '---policy eps: ', policy.epsilon

                if (i+1) / eval_args['eval_per_iter'] > last_eval_milestone:
                    last_eval_milestone = (i+1) / eval_args['eval_per_iter']
                    log += '\n'
                    log += self.eval(
                        eval_args['eval_env'], eval_args['eval_policy'],
                        eval_args['num_episodes'])

                print log
                log_file.write(log+'\n')
                log_file.flush()

                # main task ...
                t = time.time()
                state = env.reset()
                rewards = []

            state_gpu.copy_(torch.from_numpy(state.reshape(state_gpu.size())))
            action = self.select_action(state_gpu, policy)
            next_state, reward = env.step(action)

            if self.replay_memory is not None:
                self.replay_memory.append(state, action, reward, next_state, env.end)
                loss = self._update_q_net(batch_size)
            else:
                samples = [core.Sample(state, action, reward, next_state, env.end)]
                x, a, y = core.samples_to_minibatch(samples, self)
                loss = self.online_q_net.train_step(x, a, y)
            state = next_state
            losses.append(loss)
            rewards.append(reward)
            if self.use_double_q and np.random.uniform() > 0.5:
                # flip networks
                self.target_q_net, self.online_q_net = \
                    self.online_q_net, self.target_q_net
            elif self.replay_memory is None or ((i+1) % self.target_update_freq == 0):
                    self.target_q_net = copy.deepcopy(self.online_q_net)

            if (i+1) % (num_iters/4) == 0:
                model_path = os.path.join(output_path, 'net_%d.pth' % ((i+1)/(num_iters/4)))
                torch.save(self.online_q_net.state_dict(), model_path)

        torch.save(self.online_q_net.state_dict(), os.path.join(output_path, 'net_final.pth'))
        log = self.eval(eval_args['eval_env'], eval_args['eval_policy'], eval_args['num_episodes_at_end'])
        log_file.write(log+'\n')
        log_file.flush()
