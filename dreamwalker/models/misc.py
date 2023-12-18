# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last modified: 18-12-2023

# --Needed functionalities
# - 1. None

# ---DEPENDENCIES---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---RESIDUAL NETWORKS----------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, i_chan, o_chan, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.c0 = nn.Sequential(
            nn.Conv2d(i_chan, o_chan, kernel_size=1, padding=1),
            nn.BatchNorm2d(o_chan),
            nn.ReLU(),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(o_chan, o_chan, kernel_size, padding=padding),
            nn.BatchNorm2d(o_chan),
            nn.ReLU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(o_chan, o_chan, kernel_size, padding=padding),
            nn.BatchNorm2d(o_chan),
        )

    def forward(self, x):
        xci = self.c0(x)
        out = self.c1(xci)
        out = self.c2(out)
        out += xci
        out = F.relu(out, inplace=True)
        return out


class ResidualStack(nn.Module):
    def __init__(self, i_chan, o_chan, kernel_size=3, num_blocks=3):
        super(ResidualStack, self).__init__()
        self.stack = nn.ModuleList()
        for b in range(num_blocks):
            i_chan = o_chan if b > 0 else i_chan
            self.stack.append(ResidualBlock(i_chan, o_chan, kernel_size))
        self.stack = nn.Sequential(*self.stack)

    def forward(self, x):
        return self.stack(x)


# ---END------------------------------------------------------------------------
