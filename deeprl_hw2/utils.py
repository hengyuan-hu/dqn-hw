"""Common functions you may find useful in your implementation."""

# import semver
# import tensorflow as tf
import torch
import torch.nn as nn
from torch.autograd import Variable


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_zero_grads(params):
    for p in params:
        if p.grad is not None:
            utils.assert_eq(p.grad.data.sum(), 0)


def assert_frozen(module):
    for p in module.parameters():
        assert not p.requires_grad


def weights_init(m):
    """custom weights initialization called on net_g and net_f."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


# def get_uninitialized_variables(variables=None):
#     """Return a list of uninitialized tf variables.

#     Parameters
#     ----------
#     variables: tf.Variable, list(tf.Variable), optional
#       Filter variable list to only those that are uninitialized. If no
#       variables are specified the list of all variables in the graph
#       will be used.

#     Returns
#     -------
#     list(tf.Variable)
#       List of uninitialized tf variables.
#     """
#     sess = tf.get_default_session()
#     if variables is None:
#         variables = tf.global_variables()
#     else:
#         variables = list(variables)

#     if len(variables) == 0:
#         return []

#     if semver.match(tf.__version__, '<1.0.0'):
#         init_flag = sess.run(
#             tf.pack([tf.is_variable_initialized(v) for v in variables]))
#     else:
#         init_flag = sess.run(
#             tf.stack([tf.is_variable_initialized(v) for v in variables]))
#     return [v for v, f in zip(variables, init_flag) if not f]


# def get_soft_target_model_updates(target, source, tau):
#     r"""Return list of target model update ops.

#     These are soft target updates. Meaning that the target values are
#     slowly adjusted, rather than directly copied over from the source
#     model.

#     The update is of the form:

#     $W' \gets (1- \tau) W' + \tau W$ where $W'$ is the target weight
#     and $W$ is the source weight.

#     Parameters
#     ----------
#     target: keras.models.Model
#       The target model. Should have same architecture as source model.
#     source: keras.models.Model
#       The source model. Should have same architecture as target model.
#     tau: float
#       The weight of the source weights to the target weights used
#       during update.

#     Returns
#     -------
#     list(tf.Tensor)
#       List of tensor update ops.
#     """
#     pass


# def get_hard_target_model_updates(target, source):
#     """Return list of target model update ops.

#     These are hard target updates. The source weights are copied
#     directly to the target network.

#     Parameters
#     ----------
#     target: keras.models.Model
#       The target model. Should have same architecture as source model.
#     source: keras.models.Model
#       The source model. Should have same architecture as target model.

#     Returns
#     -------
#     list(tf.Tensor)
#       List of tensor update ops.
#     """
#     pass
