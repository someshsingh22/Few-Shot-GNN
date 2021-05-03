import torch.nn as nn


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
