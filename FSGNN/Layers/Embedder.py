import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, kernel_size, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
    )


class OmniglotEmbedding(nn.Module):
    """ In this network the input image is supposed to be 28x28 """

    def __init__(self, args, emb_size):
        super(OmniglotEmbedding, self).__init__()
        self.emb_size = emb_size
        self.nef = 64
        self.args = args

        # input is 1 x 28 x 28
        self.conv1 = conv3x3(1, self.nef, 3, padding=1)
        # state size. (nef) x 14 x 14
        self.conv2 = conv3x3(self.nef, self.nef, 3, padding=1)
        # state size. (1.5*ndf) x 7 x 7
        self.conv3 = conv3x3(self.nef, self.nef, 3)
        # state size. (2*ndf) x 5 x 5
        self.conv3 = conv3x3(self.nef, self.nef, 3)
        # state size. (2*ndf) x 3 x 3
        self.fc_last = nn.Linear(3 * 3 * self.nef, self.emb_size)
        self.bn_last = nn.BatchNorm1d(self.emb_size)

    def forward(self, inputs):
        e1 = F.max_pool2d(self.conv1(inputs), 2)
        x = F.leaky_relu(e1, 0.1, inplace=True)
        e2 = F.max_pool2d(self.conv2(x), 2)
        x = F.leaky_relu(e2, 0.1, inplace=True)
        e3 = self.conv3(x)
        x = F.leaky_relu(e3, 0.1, inplace=True)
        e4 = self.conv4(x)
        x = F.leaky_relu(e4, 0.1, inplace=True)
        x = x.view(-1, 3 * 3 * self.nef)
        output = F.leaky_relu(self.bn_last(self.fc_last(x)))
        return [e1, e2, e3, output]
