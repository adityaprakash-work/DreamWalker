# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last modified: 18-12-2023

# --Needed functionalities
# - 1. None

# ---DEPENDENCIES---------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F


# ---RESIDUAL NETWORKS----------------------------------------------------------
class ResidualBlock(nn.Module):
    """A simple residual block."""

    def __init__(self, n_channels, hidden_channels):
        """Initializes a new ResidualBlock instance.

        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self._net(x)


class ResidualStack(nn.Module):
    """A stack of multiple ResidualBlocks."""

    def __init__(self, n_channels, hidden_channels, n_residual_blocks=1):
        """Initializes a new ResidualStack instance.

        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
            n_residual_blocks: Number of residual blocks in the stack.
        """
        super().__init__()
        self._net = nn.Sequential(
            *[
                ResidualBlock(n_channels, hidden_channels)
                for _ in range(n_residual_blocks)
            ]
            + [nn.ReLU()]
        )

    def forward(self, x):
        return self._net(x)
