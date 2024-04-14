from typing import List
from torch.nn import Linear
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore")

import torch


class VAE(nn.Module):
    def __init__(self, input_dim: int = 1, latent_dim: int = 128, hidden_dim: List[int] = None) -> None:
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else [16, 32, 64, 128, 256]

        # Encoder definition
        encoder_modules = []
        in_channel = self.input_dim
        for dim in self.hidden_dim:
            encoder_modules.append(nn.Sequential(
                nn.Conv2d(in_channel, dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.PReLU()
            ))
            in_channel = dim

        self.encoder = nn.Sequential(*encoder_modules)

        # Latent vector representation
        self.mu = nn.Linear(self.hidden_dim[-1] * 5 * 5, self.latent_dim)  # Latent space dimension
        self.var = nn.Linear(self.hidden_dim[-1] * 5 * 5, self.latent_dim)

        # Decoder definition (reversed order of hidden dimensions)
        decoder_modules = [nn.Linear(latent_dim, self.hidden_dim[-1] * 5 * 5)]
        hidden_decoder = self.hidden_dim[::-1]  # Reverse the list

        for i in range(len(hidden_decoder) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_decoder[i], hidden_decoder[i + 1], kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_decoder[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*decoder_modules)

        # Final layer for reconstructing the input
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim[-1], self.hidden_dim[-1], kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dim[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: torch.Tensor) -> list[Linear]:
        """
        Encodes the input tensor into latent representation (mu and var).

        Args:
            input: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            A list containing two linear layers: mu and var representing the latent means and variances.
        """
        result = torch.flatten(self.encoder(input))
        mu = self.mu(result)
        var = self.var(result)

        return [mu, var]

    def reparameterize(self, mu: Linear, var: Linear) -> torch.Tensor:
        """
        Reparameterization trick for sampling from the latent distribution.

        Args:
            mu: Linear layer representing the latent means.
            var: Linear layer representing the latent variances.

        Returns:
            A tensor sampled from the latent distribution using reparameterization trick.
        """
        # Implement the reparameterization trick here (epsilon sampling)
        epsilon = torch.randn(mu.size())  # Sample random noise
        std = torch.exp(0.5 * var)  # Standard deviation from variance
        z = mu + epsilon * std
        return z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent vector into reconstructed input.

        Args:
            z: Latent vector tensor.

        Returns:
            A tensor representing the reconstructed input.
        """
        result = self.decoder(z)
        return self.final_layer(result)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu, var = self.encode(input)
        z = self.reparameterize(mu, var)
        return self.decode(z)

    def kl_divergence(self, mu, logvar):
        """
        Calculates the KL divergence between a standard normal distribution
        and the distribution encoded by the VAE.

        Args:
          mu: Tensor of mean values from the encoder (batch_size, latent_dim).
          logvar: Tensor of logarithm of variances from the encoder (batch_size, latent_dim).

        Returns:
          Tensor: KL divergence loss (batch_size).
        """
        # Epsilon for numerical stability
        epsilon = 1e-6

        # Standard normal distribution with zero mean and unit variance
        # standard_normal_dist = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))

        # KL divergence calculation (using element-wise operations)
        kl_div = torch.mean(mu ** 2 + logvar - torch.log(logvar + epsilon) - 1, dim=1)
        return kl_div

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.forward(self.decode(torch.randn(num_samples, self.latent_dim)))
