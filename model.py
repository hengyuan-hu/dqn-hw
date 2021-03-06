"""Implement the Q network as a torch.nn Module"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils


class QNetwork(nn.Module):
    def __init__(self, num_frames, frame_size, num_actions, optim_args, net_file):
        """
        num_frames: i.e. num of channels of input
        frame_size: int, frame has to be square for simplicity
        num_actions: i.e. num of output Q values
        """
        super(QNetwork, self).__init__()

        self._build_model((num_frames, frame_size, frame_size), num_actions)
        utils.init_net(self, net_file)
        self.cuda()
        self.optim = torch.optim.RMSprop(self.parameters(), **optim_args)

    def _build_model(self, input_shape, num_actions):
        """
        input_shape: (num_channel, frame_size, frame_size)
        num_actions: decides num of outputs of q_net
        """
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def loss(self, x, a, y):
        utils.assert_eq(a.dim(), 2)
        q_vals = self.forward(Variable(x))
        utils.assert_eq(q_vals.size(), a.size())
        y_pred = (q_vals * Variable(a)).sum(1)
        err = nn.functional.smooth_l1_loss(y_pred, Variable(y))
        return err

    def train_step(self, x, a, y, grad_clip=None):
        err = self.loss(x, a, y)
        err.backward()
        if grad_clip:
            nn.utils.clip_grad_norm(self.parameters(), grad_clip)
        self.optim.step()
        self.zero_grad()
        return err.data[0]


class DQNetwork(QNetwork):

    def _build_model(self, input_shape, num_actions):
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(input_shape[0], 32, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(32, 64, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))
        conv.add_module('conv3', nn.Conv2d(64, 64, 3, 1))
        conv.add_module('relu3', nn.ReLU(inplace=True))

        num_fc_in = utils.count_output_size((1,)+input_shape, conv)
        num_fc_out = 512
        fc = nn.Sequential()
        fc.add_module('fc1', nn.Linear(num_fc_in, num_fc_out))
        fc.add_module('fc_relu1', nn.ReLU(inplace=True))
        fc.add_module('output', nn.Linear(num_fc_out, num_actions))

        self.conv = conv
        self.fc = fc

    def forward(self, x):
        x.div_(255.0)
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        utils.assert_eq(y.dim(), 2)
        return y


class DuelingQNetwork(QNetwork):

    def _build_model(self, input_shape, num_actions):
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(input_shape[0], 32, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(32, 64, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))
        conv.add_module('conv3', nn.Conv2d(64, 64, 3, 1))
        conv.add_module('relu3', nn.ReLU(inplace=True))

        num_fc_in = utils.count_output_size((1,)+input_shape, conv)
        num_fc_out = 512
        fc_a = nn.Sequential()
        fc_a.add_module('fc1', nn.Linear(num_fc_in, num_fc_out))
        fc_a.add_module('fc_relu1', nn.ReLU(inplace=True))
        fc_a.add_module('adv', nn.Linear(num_fc_out, num_actions))

        fc_v = nn.Sequential()
        fc_v.add_module('fc2', nn.Linear(num_fc_in, num_fc_out))
        fc_v.add_module('fc_relu2', nn.ReLU(inplace=True))
        fc_v.add_module('val', nn.Linear(num_fc_out, 1))

        self.conv = conv
        self.fc_a = fc_a
        self.fc_v = fc_v

    def forward(self, x):
        x.div_(255.0)
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        a = self.fc_a(y)
        v = self.fc_v(y) - a.mean(1)
        y = a + v.expand_as(a)
        utils.assert_eq(y.dim(), 2)
        return y


class PredDuelingQNetwork(QNetwork):

    def _build_model(self, input_shape, num_actions):
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(input_shape[0], 32, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(32, 64, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))
        conv.add_module('conv3', nn.Conv2d(64, 64, 3, 1))
        conv.add_module('relu3', nn.ReLU(inplace=True))

        num_fc_in = utils.count_output_size((1,)+input_shape, conv)
        num_fc_out = 512
        fc_a = nn.Sequential()
        fc_a.add_module('fc_a', nn.Linear(num_fc_in, num_fc_out))
        fc_a.add_module('relu_a', nn.ReLU(inplace=True))
        fc_a.add_module('adv', nn.Linear(num_fc_out, num_actions))

        fc_v = nn.Sequential()
        fc_v.add_module('fc_v', nn.Linear(num_fc_in, num_fc_out))
        fc_v.add_module('relu_v', nn.ReLU(inplace=True))
        fc_v.add_module('val', nn.Linear(num_fc_out, 1))

        fc_pred_v = nn.Sequential()
        fc_pred_v.add_module('fc_pred_v', nn.Linear(num_fc_in, num_fc_out))
        fc_pred_v.add_module('relu_pred_v', nn.ReLU(inplace=True))
        fc_pred_v.add_module('pred_v', nn.Linear(num_fc_out, num_actions))

        self.conv = conv
        self.fc_a = fc_a
        self.fc_v = fc_v
        self.fc_pred_v = fc_pred_v

    def forward(self, x, pred):
        # TODO: better naming if works
        x.div_(255.0)
        conv = self.conv(x)
        conv = conv.view(conv.size(0), -1)
        a = self.fc_a(conv)
        a.sub_(a.mean(1).expand_as(a))
        v = self.fc_v(conv)
        q = a + v.expand_as(a)
        utils.assert_eq(q.dim(), 2)
        pred_v = None
        if pred:
            pred_v = self.fc_pred_v(conv)
            utils.assert_eq(pred_v.size(), q.size())
        return q, v, pred_v

    def loss(self, x, a, y, next_v):
        q_vals, _, pred_v = self.forward(Variable(x), pred=True)
        a = Variable(a)
        y_pred = (q_vals * a).sum(1)
        y_err = nn.functional.smooth_l1_loss(y_pred, Variable(y))

        next_v_pred = (pred_v * a).sum(1)
        next_v_err = nn.functional.smooth_l1_loss(next_v_pred, Variable(next_v))
        return y_err, next_v_err

    def train_step(self, x, a, y, next_v, grad_clip=None):
        y_err, next_v_err = self.loss(x, a, y, next_v)
        err = y_err + next_v_err
        err.backward()
        if grad_clip:
            nn.utils.clip_grad_norm(self.parameters(), grad_clip)
        self.optim.step()
        self.zero_grad()
        return y_err.data[0], next_v_err.data[0]


class SinglePredDuelingQNetwork(QNetwork):

    def _build_model(self, input_shape, num_actions):
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(input_shape[0], 32, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(32, 64, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))
        conv.add_module('conv3', nn.Conv2d(64, 64, 3, 1))
        conv.add_module('relu3', nn.ReLU(inplace=True))

        num_fc_in = utils.count_output_size((1,)+input_shape, conv)
        num_fc_out = 512
        fc_a = nn.Sequential()
        fc_a.add_module('fc_a', nn.Linear(num_fc_in, num_fc_out))
        fc_a.add_module('relu_a', nn.ReLU(inplace=True))
        fc_a.add_module('adv', nn.Linear(num_fc_out, num_actions))

        fc_v = nn.Sequential()
        fc_v.add_module('fc_v', nn.Linear(num_fc_in, num_fc_out))
        fc_v.add_module('relu_v', nn.ReLU(inplace=True))
        fc_v.add_module('val', nn.Linear(num_fc_out, 1))

        # fc_pred_v = nn.Sequential()
        self.pred_v_fc1 = nn.Linear(num_fc_in, num_fc_out)
        self.pred_v_relu1 = nn.ReLU(inplace=True)
        self.pred_v_output = nn.Linear(num_fc_out+1, 1)

        self.conv = conv
        self.fc_a = fc_a
        self.fc_v = fc_v

    def forward(self, x, pred):
        # TODO: better naming if works
        x.div_(255.0)
        conv = self.conv(x)
        conv = conv.view(conv.size(0), -1)
        a = self.fc_a(conv)
        a.sub_(a.mean(1).expand_as(a))
        v = self.fc_v(conv)
        q = a + v.expand_as(a)
        utils.assert_eq(q.dim(), 2)
        pred_v = None
        if pred:
            action = a.max(1)[1].data.float()
            # print action[:10]
            pred_v = nn.functional.relu(self.pred_v_fc1(conv))
            pred_v = torch.cat((pred_v, Variable(action)), 1)
            # print pred_v.size()
            pred_v = self.pred_v_output(pred_v)
            # print '>>>', pred_v.size()
            # pred_v = self.fc_pred_v(conv)
            # utils.assert_eq(pred_v.size(), q.size())
        return q, v, pred_v

    def loss(self, x, a, y, next_v):
        q_vals, _, pred_v = self.forward(Variable(x), pred=True)
        a = Variable(a)
        y_pred = (q_vals * a).sum(1)
        y_err = nn.functional.smooth_l1_loss(y_pred, Variable(y))
        # next_v_pred = (pred_v * a).sum(1)
        next_v_err = nn.functional.smooth_l1_loss(pred_v, Variable(next_v))
        return y_err, next_v_err

    def train_step(self, x, a, y, next_v, grad_clip=None):
        y_err, next_v_err = self.loss(x, a, y, next_v)
        err = y_err + next_v_err
        err.backward()
        if grad_clip:
            nn.utils.clip_grad_norm(self.parameters(), grad_clip)
        self.optim.step()
        self.zero_grad()
        return y_err.data[0], next_v_err.data[0]


if __name__ == '__main__':
    import copy

    qn = QNetwork(4, 84, 4, 0.1)
    print qn
    for p in qn.parameters():
        print p.mean().data[0], p.std().data[0]
    fake_input = Variable(torch.cuda.FloatTensor(10, 4, 84, 84), volatile=True)
    print qn(fake_input).size()
    qn_target = copy.deepcopy(qn)
