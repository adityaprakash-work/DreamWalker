# trunk-ignore-all(black)
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F

import torcheeg as tg
from torcheeg.models.transformer.conformer import (
    PatchEmbedding,
    TransformerEncoder,
)

from .misc import ResidualStack


# ---CONFORMER EEG ENCODER------------------------------------------------------
class ConformerEEGEncoder(nn.Module):
    def __init__(
        self,
        num_electrodes: int = 128,
        num__samples: int = 500,
        sampling_rate: int = 100,
        embed_dropout: float = 0.5,
        hid_channels: int = 40,
        depth: int = 6,
        heads: int = 10,
        dropout: float = 0.5,
        forward_expansion: int = 4,
        forward_dropout: float = 0.5,
        blat_shape: tuple = (16, 64, 54),
        tlat_shape: tuple = (16, 32, 32),
        num_res_blocks: int = 3,
    ):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.num__samples = num__samples
        self.sampling_rate = sampling_rate
        self.embed_dropout = embed_dropout
        self.hid_channels = hid_channels
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        self.forward_dropout = forward_dropout
        self.blat_shape = blat_shape
        self.tlat_shape = tlat_shape
        self.num_res_blocks = num_res_blocks

        self.emb = PatchEmbedding(num_electrodes, hid_channels, embed_dropout)
        self.enc = TransformerEncoder(
            depth,
            hid_channels,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            forward_dropout=forward_dropout,
        )

        entr_units = self.get_mock_shape()
        entr_units = entr_units[1] * entr_units[2]
        b_exit_units = blat_shape[0] * blat_shape[1] * blat_shape[2]
        b_betw_units = int((entr_units + b_exit_units) / 2)
        t_exit_units = tlat_shape[0] * tlat_shape[1] * tlat_shape[2]
        t_betw_units = int((b_exit_units + t_exit_units) / 2)

        # Correction FC Bottom
        self.cfcb = nn.Sequential(
            nn.Linear(entr_units, b_betw_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(b_betw_units, b_exit_units),
        )
        # Residual Stack Bottom
        self.rstb = ResidualStack(
            self.blat_shape[0], self.blat_shape[0], self.num_res_blocks
        )

        # Correction FC Top
        self.cfct = nn.Sequential(
            nn.Linear(b_exit_units, t_betw_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(t_betw_units, t_exit_units),
        )
        # Residual Stack Top
        self.rstt = ResidualStack(
            self.tlat_shape[0], self.tlat_shape[0], self.num_res_blocks
        )

    def get_mock_shape(self):
        x = torch.rand(1, 1, self.num_electrodes, self.num__samples)
        x = self.emb(x)
        x = self.enc(x)
        return x.shape

    def forward(self, x):
        x = self.emb(x)
        x = self.enc(x)
        x = x.view(x.shape[0], -1)
        xb = self.cfcb(x)
        xb = xb.view(xb.shape[0], *self.blat_shape)
        xb = self.rstb(xb)
        xt = xb.view(xb.shape[0], -1)
        xt = self.cfct(xt)
        xt = xt.view(xt.shape[0], *self.tlat_shape)
        xt = self.rstt(xt)
        return xb, xt
