"""Implement the Q network as a torch.nn Module"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils


class QNetwork(nn.Module):
    def __init__(self, num_frames, frame_size, num_actions):
        """
        num_frames: i.e. num of channels of input
        frame_size: int, frame has to be square for simplicity
        num_actions: i.e. num of output Q values
        """
        super(QNetwork, self).__init__()

        # TODO: padding or not???
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(num_frames, 16, 8, 4))
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(16, 32, 4, 2))
        conv.add_module('relu2', nn.ReLU(inplace=True))

        fake_input = Variable(
            torch.FloatTensor(1, num_frames, frame_size, frame_size),
            volatile=True)
        num_fc_in = conv.forward(fake_input).view(-1).size()[0]
        fc = nn.Sequential()
        fc.add_module('fc1', nn.Linear(num_fc_in, 256))
        fc.add_module('fc_relu1', nn.ReLU(inplace=True))
        fc.add_module('output', nn.Linear(256, num_actions))

        self.conv = conv
        self.fc = fc
        utils.weights_init(self)
        self.cuda()

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        utils.assert_eq(y.dim(), 2)
        return y


if __name__ == '__main__':
    qn = QNetwork(4, 84, 4)
    print qn
    fake_input = Variable(torch.cuda.FloatTensor(10, 4, 84, 84), volatile=True)
    print qn(fake_input).size()
