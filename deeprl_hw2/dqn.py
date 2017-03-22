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
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in):
        self.online_q_net = q_network
        self.target_q_net = copy.deepcopy(q_network)

        self.replay_memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in

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

    def _online_q_values(self, states):
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
        q_vals = self._online_q_values(states)
        utils.assert_eq(q_vals.size()[0], 1)
        q_vals = q_vals.view(q_vals.size()[1])
        action = policy(q_vals.cpu().numpy())
        return action

    def _update_q_net(self, batch_size):
        samples = self.replay_memory.sample(batch_size)
        x, a, y = core.samples_to_minibatch(samples, self)
        loss = self.online_q_net.train_step(x, a, y)
        return loss

    def train(self, env, policy, batch_size, num_iters, log_file, eval_args):
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
        state_gpu = torch.cuda.FloatTensor(
            1, env.num_frames, env.frame_size, env.frame_size)
        state = self._burn_in(env)

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

                if num_episodes % eval_args['eval_per_eps'] == 0:
                    log += '\n'
                    log += self.eval(
                        env, eval_args['eval_policy'], eval_args['num_episodes'])

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

            if (i+1) % self.target_update_freq:
                self.target_q_net = copy.deepcopy(self.online_q_net)

    def eval(self, env, policy, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        state_gpu = torch.cuda.FloatTensor(
            1, env.num_frames, env.frame_size, env.frame_size)
        state = env.reset() # TODO: em???

        total_rewards = np.zeros(num_episodes)
        rewards = []
        eps_idx = 0
        log = ''
        while eps_idx < num_episodes:
            if env.end:
                state = env.reset()
                total_rewards[eps_idx] = sum(rewards)
                eps_log = ('>>>Eval: [%d/%d], rewards: %s\n' %
                           (eps_idx+1, num_episodes, total_rewards[eps_idx]))
                log += eps_log
                eps_idx += 1
                rewards = []

            state_gpu.copy_(torch.from_numpy(state.reshape(state_gpu.size())))
            action = self.select_action(state_gpu, policy)
            state, reward = env.step(action)
            rewards.append(reward)

        eps_log = '>>>Eval: avg total rewards: %s' % total_rewards.mean()
        log += eps_log
        return log
