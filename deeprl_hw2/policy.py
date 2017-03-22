"""RL Policy classes.

We have provided you with a base policy class, some example
implementations and some unimplemented classes that should be useful
in your code.
"""
import numpy as np
import utils


class GreedyEpsilonPolicy(object):
    def __init__(self, epsilon):
        self.epsilon = np.float32(epsilon)

    def __call__(self, q_values):
        return self.select_action(q_values)

    def _greedy(self):
        return np.random.uniform() > self.epsilon

    def select_action(self, q_values):
        """Run Greedy-Epsilon for the given Q-values.
        q_values: 1-d numpy.array
        return: action, int
        """
        utils.assert_eq(len(q_values.shape), 1)
        if self._greedy():
            action = q_values.argmax()
        else:
            num_actions = q_values.shape[0]
            action = np.random.randint(0, num_actions)
        return action


class LinearDecayGreedyEpsilonPolicy(GreedyEpsilonPolicy):
    """Policy with a parameter that decays linearly.

    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.

    Parameters
    ----------
    start_value: int, float
      The initial value of the parameter
    end_value: int, float
      The value of the policy at the end of the decay.
    num_steps: int
      The number of steps over which to decay the value.

    """

    def __init__(self, start_eps, end_eps, num_steps):
        super(LinearDecayGreedyEpsilonPolicy, self).__init__(start_eps)
        self.num_steps = num_steps
        self.decay_rate = (start_eps - end_eps) / float(num_steps)

    def select_action(self, q_values):
        action = super(LinearDecayGreedyEpsilonPolicy, self).select_action(q_values)
        if self.num_steps > 0:
            self.epsilon -= self.decay_rate
            self.num_steps -= 1
        return action


if __name__ == '__main__':
    q_values = np.random.uniform(0, 1, (3,))
    target_actions = q_values.argmax()

    greedy_policy = GreedyEpsilonPolicy(0)
    actions = greedy_policy(q_values)
    assert (actions == target_actions).all()

    uniform_policy = GreedyEpsilonPolicy(1)
    uni_actions = uniform_policy(q_values)
    assert not (uni_actions == target_actions).all()

    steps = 9
    ldg_policy = LinearDecayGreedyEpsilonPolicy(1, 0.1, steps)
    expect_eps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1]
    actual_eps = [1.0]
    for i in range(steps+1):
        actions = ldg_policy(q_values)
        actual_eps.append(ldg_policy.epsilon)
    assert (np.abs((np.array(actual_eps) - np.array(expect_eps)))
            < 1e-5).all()
