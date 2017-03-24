"""Implement the Q network as a torch.nn Module"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils


class QNetwork(nn.Module):
    def __init__(self, num_frames, frame_size, num_actions, update_freq, optim_args, net_file=None):
        """
        num_frames: i.e. num of channels of input
        frame_size: int, frame has to be square for simplicity
        num_actions: i.e. num of output Q values
        """
        super(QNetwork, self).__init__()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_actions = num_actions
        self.update_freq = update_freq
        self.optim_args = optim_args
        self.net_file = net_file

        self.step = 0

        self._build_model()

    def _build_model(self):
        return

    def forward(self, x):
        return x

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
    def __init__(self, num_frames, frame_size, num_actions, update_freq, optim_args, net_file=None):
        super(DQNetwork, self).__init__(num_frames, frame_size, num_actions,
                                        update_freq, optim_args, net_file)

    def _build_model(self):
        # TODO: padding or not???
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(self.num_frames, 16, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(16, 32, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))

        # TODO: factor this out as helper
        fake_input = Variable(
            torch.FloatTensor(1, self.num_frames, self.frame_size, self.frame_size),
            volatile=True)
        num_fc_in = conv.forward(fake_input).view(-1).size()[0]

        num_fc_out = 256
        fc = nn.Sequential()
        fc.add_module('fc1', nn.Linear(num_fc_in, num_fc_out))
        fc.add_module('fc_relu1', nn.ReLU(inplace=True))

        q_net = nn.Sequential()
        q_net.add_module('output', nn.Linear(num_fc_out, self.num_actions))

        predictor = nn.Sequential()
        predictor.add_module('pd1', nn.Linear(num_fc_out, num_fc_out/2))
        predictor.add_module('pd_relu1', nn.ReLU(inplace=True))
        predictor.add_module('pd2', nn.Linear(num_fc_out/2, num_fc_out))
        predictor.add_module('pd_relu2', nn.ReLU(inplace=True))

        self.conv = conv
        self.fc = fc
        self.q_net = q_net
        self.predictor = predictor

        # TODO: move this out
        utils.init_net(self, self.net_file)
        self.cuda()
        self.optim = torch.optim.RMSprop(self.parameters(), **self.optim_args)

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
    def __init__(self, num_frames, frame_size, num_actions,
                 update_freq, optim_args, net_file=None):
        super(DeeperQNetwork, self).__init__(
            num_frames, frame_size, num_actions,
            update_freq, optim_args, net_file)

    def _build_model(self):
        # TODO: padding or not???
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(self.num_frames, 32, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(32, 64, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))
        conv.add_module('conv3', nn.Conv2d(64, 64, 3, 1))
        conv.add_module('relu3', nn.ReLU(inplace=True))

        fake_input = Variable(
            torch.FloatTensor(1, self.num_frames, self.frame_size,
                              self.frame_size), volatile=True)
        num_fc_in = conv.forward(fake_input).view(-1).size()[0]
        fc = nn.Sequential()
        fc.add_module('fc1', nn.Linear(num_fc_in, 512))
        fc.add_module('fc_relu1', nn.ReLU(inplace=True))
        fc.add_module('output', nn.Linear(512, self.num_actions))

        self.conv = conv
        self.fc = fc
        utils.init_net(self, self.net_file)
        self.cuda()
        self.optim = torch.optim.RMSprop(self.parameters(), **self.optim_args)


class DuelingQNetwork(QNetwork):
    def __init__(self, num_frames, frame_size, num_actions, update_freq, optim_args, net_file=None):
        super(DuelingQNetwork, self).__init__(num_frames, frame_size, num_actions,
                                             update_freq, optim_args, net_file)

    def _build_model(self):
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(self.num_frames, 16, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(16, 32, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))

        fake_input = Variable(
            torch.FloatTensor(1, self.num_frames, self.frame_size,
                              self.frame_size), volatile=True)
        num_fc_in = conv.forward(fake_input).view(-1).size()[0]

        fc_a =  nn.Sequential()
        fc_a.add_module('fc1', nn.Linear(num_fc_in, 256))
        fc_a.add_module('fc_relu1', nn.ReLU(inplace=True))
        fc_a.add_module('advantages', nn.Linear(256, self.num_actions))

        fc_v = nn.Sequential()
        fc_v.add_module('fc2', nn.Linear(num_fc_in, 256))
        fc_v.add_module('fc_relu2', nn.ReLU(inplace=True))
        fc_v.add_module('value', nn.Linear(256, 1))

        self.conv = conv
        self.fc_a = fc_a
        self.fc_v = fc_v

        utils.init_net(self, self.net_file)
        self.cuda()
        self.optim = torch.optim.RMSprop(self.parameters(), **self.optim_args)

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        a = self.fc_a(y)
        a.sub_(a.mean(1).expand_as(a))
        v = self.fc_v(y).expand_as(a)
        y = a + v
        utils.assert_eq(y.dim(), 2)
        return y


class LinearQNetwork(QNetwork):
    def __init__(self, num_frames, frame_size, num_actions, update_freq, optim_args, net_file=None):
        super(LinearQNetwork, self).__init__(num_frames, frame_size, num_actions,
                                             update_freq, optim_args, net_file)

    def _build_model(self):
        num_inputs = self.num_frames * self.frame_size * self.frame_size
        self.fc = nn.Linear(num_inputs, self.num_actions)
        utils.init_net(self, self.net_file)
        self.cuda()
        self.optim = torch.optim.RMSprop(self.parameters(), **self.optim_args)

    def forward(self, x):
        y = x.view(x.size(0), -1)
        y = self.fc(y)
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
