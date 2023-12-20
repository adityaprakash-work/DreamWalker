# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last modified: 19-12-2023

# --Needed functionalities
# - 1. Progressive VQVAE model

# ---DEPENDENCIES---------------------------------------------------------------
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from .misc import ResidualStack
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


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
        self.net.append(nn.ReLU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        return x


# ---VECTOR QUANTIZER-----------------------------------------------------------
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
class ResVQVAE(nn.Module):
    def __init__(
        self,
        enc_i_chan=3,
        enc_o_chan=64,
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
        super(ResVQVAE, self).__init__()
        self.encoder = ResEncoder(
            enc_i_chan,
            enc_o_chan,
            num_pools,
            num_res_blocks,
            latent_activation,
        )
        self.quantizer = Quantizer(num_emb, emb_dim, comm_cost, beta)
        self.decoder = ResDecoder(
            enc_i_chan,
            enc_o_chan,
            num_ups,
            num_res_blocks,
            upsampling_type,
        )

    def forward(self, x):
        z_e = self.encoder(x)
        c_loss, z_q = self.quantizer(z_e)
        recon = self.decoder(z_q)
        rec_loss = F.mse_loss(recon, x)
        return c_loss, rec_loss, z_e, z_q, recon


# ---TRAINERS-------------------------------------------------------------------
class VQVAETrainer:
    def __init__(
        self,
        model=None,
        optimizer=None,
        train_loader=None,
        valid_loader=None,
        metrics=None,
        device=None,
        load_checkpoint_path=None,
    ):
        if load_checkpoint_path is not None:
            self.model, self.optimizer, self.prev_logs = self.load_checkpoint(
                load_checkpoint_path
            )
        else:
            self.model = model
            self.optimizer = optimizer
            self.logs = {
                "Tr": {"loss": [], "c_loss": [], "rec_loss": []},
                "Vl": {"loss": [], "c_loss": [], "rec_loss": []},
            }
            self.metrics = metrics
            if self.metrics is not None:
                for m in self.metrics:
                    self.logs["train"][m] = []
                    self.logs["valid"][m] = []

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

    def save_checkpoint(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "logs": self.logs,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return self.model, self.optimizer, checkpoint["logs"]

    def run_on_loader(self, epoch_idx, phase="Tr"):
        log = {k: None for k in self.logs[phase].keys()}
        if phase == "Tr":
            self.model.train()
            loader = self.train_loader
        elif phase == "Vl":
            self.model.eval()
            loader = self.valid_loader
        pbar = tqdm(loader, desc=f"Epoch {epoch_idx + 1} | {phase}")
        for batch_idx, (data, _) in enumerate(pbar):
            if phase == "Tr":
                data = data.to(self.device)
                self.optimizer.zero_grad()
                c_loss, rec_loss, z_e, z_q, recon = self.model(data)
                loss = c_loss + rec_loss
                loss.backward()
                self.optimizer.step()
            elif phase == "Vl":
                with torch.no_grad():
                    data = data.to(self.device)
                    c_loss, rec_loss, z_e, z_q, recon = self.model(data)
                    loss = c_loss + rec_loss

            log["loss"] = (
                loss.item() / len(loader) + log["loss"]
                if log["loss"] is not None
                else loss.item() / len(loader)
            )
            log["c_loss"] = (
                c_loss.item() / len(loader) + log["c_loss"]
                if log["c_loss"] is not None
                else c_loss.item() / len(loader)
            )
            log["rec_loss"] = (
                rec_loss.item() / len(loader) + log["rec_loss"]
                if log["rec_loss"] is not None
                else rec_loss.item() / len(loader)
            )
            if self.metrics is not None:
                for m in self.metrics:
                    mval = self.metrics[m](self.model, data).item()
                    log[m] = (
                        mval / len(loader) + log[m]
                        if log[m] is not None
                        else mval / len(loader)
                    )
            pbar.set_postfix(log)

        return log, (data, z_e, z_q, recon)

    def train(self, epochs, monitors, plot=True):
        for epoch_idx in range(epochs):
            log, _ = self.run_on_loader(epoch_idx, phase="Tr")
            for k in log.keys():
                self.logs["Tr"][k].append(log[k])
            for m in monitors:
                halt, _ = m(self, epoch_idx, "Tr")
                if halt:
                    return
            log, (d, e, q, r) = self.run_on_loader(epoch_idx, phase="Vl")
            for k in log.keys():
                self.logs["Vl"][k].append(log[k])
            for m in monitors:
                halt, _ = m(self, epoch_idx, "Vl")
                if halt:
                    return

            if plot:
                plot_vqvae_forward(d, e, q, r)


# ---PLOTTING-------------------------------------------------------------------
def plot_vqvae_forward(data, z_e, z_q, recon):
    data_grid = make_grid(data.cpu().detach(), nrow=8)
    recon_grid = make_grid(recon.cpu().detach(), nrow=8)
    recon_grid = (recon_grid - recon_grid.min()) / (recon_grid.max() - recon_grid.min())

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(data_grid.permute(1, 2, 0))
    ax[0, 0].set_title("Original Images")
    ax[0, 1].imshow(recon_grid.permute(1, 2, 0))
    ax[0, 1].set_title("Reconstructed Images")

    pca = PCA(n_components=2)
    z_e = z_e.view(z_e.shape[0], -1)
    z_q = z_q.view(z_q.shape[0], -1)
    z_e = pca.fit_transform(z_e.cpu().detach().numpy())
    z_q = pca.transform(z_q.cpu().detach().numpy())

    # Plot the encoded representation
    ax[1, 0].scatter(z_e[:, 0], z_e[:, 1], c="r", s=2)
    ax[1, 0].set_title("Encoded Representation")

    # Plot the encoded + quantized representation
    ax[1, 1].scatter(z_e[:, 0], z_e[:, 1], c="r", s=2)
    ax[1, 1].scatter(z_q[:, 0], z_q[:, 1], c="b", s=2)
    ax[1, 1].set_title("Encoded + Quantized Representation")

    plt.show()


# ---END------------------------------------------------------------------------
