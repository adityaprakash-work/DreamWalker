"""Models available in pytorch-generative."""

from ..models.autoregressive.fvbn import FullyVisibleBeliefNetwork
from ..models.autoregressive.gated_pixel_cnn import GatedPixelCNN
from ..models.autoregressive.image_gpt import ImageGPT
from ..models.autoregressive.made import MADE
from ..models.autoregressive.nade import NADE
from ..models.autoregressive.pixel_cnn import PixelCNN
from ..models.autoregressive.pixel_snail import PixelSNAIL
from ..models.flow.nice import NICE
from ..models.kde import (
    GaussianKernel,
    KernelDensityEstimator,
    ParzenWindowKernel,
)
from ..models.mixture_models import (
    BernoulliMixtureModel,
    GaussianMixtureModel,
)
from ..models.vae.beta_vae import BetaVAE
from ..models.vae.vae import VAE
from ..models.vae.vd_vae import VeryDeepVAE
from ..models.vae.vq_vae import VectorQuantizedVAE
from ..models.vae.vq_vae_2 import VectorQuantizedVAE2
