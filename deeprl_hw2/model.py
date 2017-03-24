"""Implement the Q network as a torch.nn Module"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils


class QNetwork(nn.Module):
    def __init__(self, num_frames, frame_size, num_actions,
                 update_freq, optim_args, net_file):
        """
        num_frames: i.e. num of channels of input
        frame_size: int, frame has to be square for simplicity
        num_actions: i.e. num of output Q values
        """
        super(QNetwork, self).__init__()
        self.update_freq = update_freq
        self.step = 0

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
        err = self.loss_func(y_pred, Variable(y))
        return err

    def train_step(self, x, a, y):
        """accum grads and apply every update_freq
           equivalent to augmenting batch_size by a factor of update_freq
        """
        self.step = (self.step + 1) % self.update_freq
        err = self.loss(x, a, y) # / self.update_freq
        err.backward()

        if self.step == 0:
            self.optim.step()
            self.zero_grad()
        return err.data[0]


class DQNetwork(QNetwork):

    def _build_model(self, input_shape, num_actions):
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(input_shape[0], 16, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(16, 32, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))

        num_fc_in = utils.count_output_size((1,)+input_shape, conv)
        num_fc_out = 256
        fc = nn.Sequential()
        fc.add_module('fc1', nn.Linear(num_fc_in, num_fc_out))
        fc.add_module('fc_relu1', nn.ReLU(inplace=True))

        q_net = nn.Sequential()
        q_net.add_module('output', nn.Linear(num_fc_out, num_actions))

        predictor = nn.Sequential()
        predictor.add_module('pd1', nn.Linear(num_fc_out, num_fc_out/2))
        predictor.add_module('pd_relu1', nn.ReLU(inplace=True))
        predictor.add_module('pd2', nn.Linear(num_fc_out/2, num_fc_out))
        predictor.add_module('pd_relu2', nn.ReLU(inplace=True))

        self.conv = conv
        self.fc = fc
        self.q_net = q_net
        self.predictor = predictor

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        utils.assert_eq(feat.dim(), 2)

        q_val = self.q_net(feat)
        pred_feat = self.predictor(feat)
        return q_val, feat, pred_feat

    def loss(self, x, a, y, next_feat):
        utils.assert_eq(a.dim(), 2)
        q_vals, _, pred_feat = self.forward(Variable(x))

        utils.assert_eq(q_vals.size(), a.size())
        y_pred = (q_vals * Variable(a)).sum(1)
        y_err = nn.functional.smooth_l1_loss(y_pred, Variable(y))

        utils.assert_eq(pred_feat.size(), next_feat.size())
        pred_err = nn.functional.smooth_l1_loss(pred_feat, Variable(next_feat))
        return y_err, pred_err

    def train_step(self, x, a, y, next_feat):
        """accum grads and apply every update_freq
           equivalent to augmenting batch_size by a factor of update_freq
        """
        self.step = (self.step + 1) % self.update_freq
        y_err, pred_err = self.loss(x, a, y, next_feat) # / self.update_freq
        err = y_err + pred_err
        err.backward()

        if self.step == 0:
            self.optim.step()
            self.zero_grad()
        return y_err.data[0], pred_err.data[0]


class DeeperQNetwork(DQNetwork):

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


class DuelingQNetwork(QNetwork):

    def _build_model(self, input_shape, num_actions):
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(input_shape[0], 16, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(16, 32, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))

        num_fc_in = utils.count_output_size((1,)+input_shape, conv)
        fc_a =  nn.Sequential()
        fc_a.add_module('fc1', nn.Linear(num_fc_in, 256))
        fc_a.add_module('fc_relu1', nn.ReLU(inplace=True))
        fc_a.add_module('advantages', nn.Linear(256, num_actions))

        fc_v = nn.Sequential()
        fc_v.add_module('fc2', nn.Linear(num_fc_in, 256))
        fc_v.add_module('fc_relu2', nn.ReLU(inplace=True))
        fc_v.add_module('value', nn.Linear(256, 1))

        self.conv = conv
        self.fc_a = fc_a
        self.fc_v = fc_v

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        a = self.fc_a(y)
        a.sub_(a.mean(1).expand_as(a))
        v = self.fc_v(y).expand_as(a)
        y = a + v
        utils.assert_eq(y.dim(), 2)
        return y


if __name__ == '__main__':
    import copy

    qn = QNetwork(4, 84, 4, 0.1)
    print qn
    for p in qn.parameters():
        print p.mean().data[0], p.std().data[0]
    fake_input = Variable(torch.cuda.FloatTensor(10, 4, 84, 84), volatile=True)
    print qn(fake_input).size()
    qn_target = copy.deepcopy(qn)
