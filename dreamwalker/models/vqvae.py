# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last modified: 19-12-2023

# --Needed functionalities
# - 1. Progressive VQVAE model

# ---DEPENDENCIES---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import ResidualStack


# ---ENCODERS-------------------------------------------------------------------
class ResEncoder(nn.Module):
    def __init__(
        self,
        i_chan,
        o_chan,
        num_pools=4,
        num_res_blocks=3,
        latent_activation=nn.Tanh(),
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
                nn.Conv2d(o_chan * (num_pools + 1), o_chan, 1),
                nn.BatchNorm2d(o_chan),
                latent_activation,
            )
        )
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


# ---DECODERS-------------------------------------------------------------------
class ResDecoder(nn.Module):
    def __init__(
        self,
        enc_i_chan,
        enc_o_chan,
        num_ups=4,
        num_res_blocks=3,
        upsampling_type="conv_transpose",
    ):
        super(ResDecoder, self).__init__()

        self.net = nn.ModuleList()
        self.net.append(
            nn.Sequential(
                nn.Conv2d(enc_o_chan, enc_o_chan * (num_ups + 1), 1),
                nn.BatchNorm2d(enc_o_chan * (num_ups + 1)),
                nn.ReLU(),
            )
        )

        for i in range(num_ups):
            self.net.append(
                ResidualStack(
                    enc_o_chan * (num_ups - i + 1),
                    enc_o_chan * (num_ups - i),
                    num_blocks=num_res_blocks,
                )
            )
            if upsampling_type == "upsample":
                self.net.append(nn.Upsample(scale_factor=2))
            elif upsampling_type == "conv_transpose":
                self.net.append(
                    nn.ConvTranspose2d(
                        enc_o_chan * (num_ups - i),
                        enc_o_chan * (num_ups - i),
                        2,
                        stride=2,
                    )
                )

        self.net.append(nn.Conv2d(enc_o_chan, enc_i_chan, 1))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        return x


# ---VECTOR QUANTIZER-----------------------------------------------------------
# Vector Quantization
class Quantizer(nn.Module):
    def __init__(self, num_emb, emb_dim, comm_cost, beta=0.25):
        super(Quantizer, self).__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim  # Same as channels in encoded image
        self.comm_cost = comm_cost
        self.emb = nn.Embedding(self.num_emb, self.emb_dim)
        self.emb.weight.data.uniform_(-1 / self.num_emb, 1 / self.num_emb)
        self.beta = beta

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        flatten = x.view(-1, self.emb_dim)
        distancs = torch.cdist(flatten, self.emb.weight)
        encoding_indices = torch.argmin(distancs, dim=1)
        quantized = F.embedding(encoding_indices, self.emb.weight).view(x_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), x) * self.beta
        q_latent_loss = F.mse_loss(quantized, x.detach())
        c_loss = q_latent_loss + self.comm_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return c_loss, quantized


# ---VQVAE----------------------------------------------------------------------
class VQVAE(nn.Module):
    def __init__(
        self,
        i_chan=3,
        o_chan=64,
        num_pools=4,
        num_res_blocks=3,
        latent_activation=nn.Tanh(),
        num_ups=4,
        upsampling_type="conv_transpose",
        num_emb=512,
        emb_dim=64,
        comm_cost=0.25,
        beta=0.25,
    ):
        super(VQVAE, self).__init__()
        self.encoder = ResEncoder(
            i_chan,
            o_chan,
            num_pools,
            num_res_blocks,
            latent_activation,
        )
        self.quantizer = Quantizer(num_emb, emb_dim, comm_cost, beta)
        self.decoder = ResDecoder(
            i_chan,
            o_chan,
            num_ups,
            num_res_blocks,
            upsampling_type,
        )

    def forward(self, x):
        z = self.encoder(x)
        c_loss, quantized = self.quantizer(z)
        recon = self.decoder(quantized)
        rec_loss = F.mse_loss(recon, x)
        total_loss = rec_loss + c_loss
        return c_loss, rec_loss, total_loss, recon


# ---END------------------------------------------------------------------------
