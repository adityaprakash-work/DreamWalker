import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

from torcheeg.models.transformer.conformer import (
    PatchEmbedding,
    TransformerEncoder,
)

from ..pytorch_generative.models.vae.vaes import Encoder, ResidualStack


# ---CONFORMER EEG ENCODER------------------------------------------------------
class ConformerEEGEncoder(nn.Module):
    def __init__(
        self,
        num_electrodes: int = 128,
        num__samples: int = 500,
        sampling_rate: int = 100,
        embed_dropout: float = 0.5,
        hid_channels: int = 64,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.5,
        forward_expansion: int = 4,
        forward_dropout: float = 0.5,
        blat_shape: tuple = (16, 64, 64),
        tlat_shape: tuple = (16, 32, 32),
        num_res_blocks: int = 4,
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
        mock_x = torch.randn(1, 1, num_electrodes, num__samples)
        with torch.no_grad():
            self.emb_shape = self.emb(mock_x).shape
        self.linear = nn.Linear(
            self.emb_shape[1] * self.emb_shape[2], blat_shape[1] * blat_shape[2]
        )
        self.tran = TransformerEncoder(
            depth,
            hid_channels,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            forward_dropout=forward_dropout,
        )
        self.enc_b = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=blat_shape[0],
                kernel_size=3,
                padding=1,
            ),
            ResidualStack(blat_shape[0], blat_shape[0], num_res_blocks),
        )
        self.enc_t = Encoder(
            in_channels=blat_shape[0],
            out_channels=tlat_shape[0],
            hidden_channels=tlat_shape[0] * 2,
            n_residual_blocks=num_res_blocks,
            residual_channels=16,
            stride=2,
        )

    def forward(self, x):
        z = self.emb(x)
        z = z.flatten(1)
        z = self.linear(z)
        z = z.reshape(z.shape[0], self.blat_shape[1], self.blat_shape[2])
        z = self.tran(z)
        z = z.unsqueeze(1)
        zb = self.enc_b(z)
        zt = self.enc_t(zb)
        return zb, zt
