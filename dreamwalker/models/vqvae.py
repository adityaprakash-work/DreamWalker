# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last modified: 17-12-2023

# --Needed functionalities
# - 1. None

# ---DEPENDENCIES---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import ResidualStack


# ---ENCODERS-------------------------------------------------------------------
# Convolutional Encoder: Encodes the input to whatever latent space size is
# required c x h x w by alternating between Convolution + MaxPool layers and
# Residual Stacks. The input size is given as input_shape = (c, h, w) and the
# then model will calculate required number of conv + maxpools using a sensible
# heuristic and alternate them with Residual stacks.
class ResEncoder(nn.Module):
    def __init__(
        self,
        i_chan,
        o_chan,
        num_pools=4,
        num_res_blocks=3,
        latent_activation=nn.Sigmoid(),
    ):
        super(ResEncoder, self).__init__()
        self.net = nn.ModuleList()
        for i in range(num_pools):
            self.net.append(
                ResidualStack(
                    i_chan if i == 0 else o_chan * (i + 1),
                    o_chan * (i + 2),
                    num_blocks=num_res_blocks,
                )
            )
            self.net.append(nn.MaxPool2d(2, stride=2))

        self.net.append(
            nn.Sequential(
                nn.Conv2d(o_chan * (num_pools + 1), o_chan, 3, padding=1),
                nn.BatchNorm2d(o_chan),
                latent_activation,
            )
        )
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)
