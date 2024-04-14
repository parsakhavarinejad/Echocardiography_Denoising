from typing import List, Tuple, Any
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

    @staticmethod
    def reparameterize(mu: Linear, var: Linear) -> torch.Tensor:
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

    def forward(self, input: torch.Tensor) -> tuple[Any, Any, Any]:
        mu, var = self.encode(input)
        z = self.reparameterize(mu, var)
        return self.decode(z), mu, var

    @staticmethod
    def kl_divergence(mu, logvar, beta):
        """
        Calculates the KL divergence with a weight factor.

        Args:
            mu: Tensor of mean values from the encoder (batch_size, latent_dim).
            logvar: Tensor of logarithm of variances from the encoder (batch_size, latent_dim).
            beta: Weight factor for KL divergence (scalar).

        Returns:
            Tensor: KL divergence loss (batch_size).
        """
        # Epsilon for numerical stability
        epsilon = 1e-6

        # Standard normal distribution with zero mean and unit variance
        standard_normal_dist = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(logvar))

        # KL divergence calculation with weight
        kl_div = beta * torch.mean(mu ** 2 + logvar - torch.log(logvar + epsilon) - 1, dim=1)
        return kl_div

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor, mu: Linear, logvar: Linear,
             reconstruction_criterion: nn.MSELoss = nn.MSELoss()) -> int:
        """
        Combined loss function for VAE with reconstruction and KL divergence.

        Args:
          outputs: Reconstructed image tensor from the VAE (batch_size, channels, height, width).
          targets: Ground truth image tensor (batch_size, channels, height, width).
          mu: Tensor of mean values from the encoder (batch_size, latent_dim).
          logvar: Tensor of logarithm of variances from the encoder (batch_size, latent_dim).
          reconstruction_criterion: Reconstruction loss function (default: nn.MSELoss()).

        Returns:
          Tensor: Combined loss (scalar).
        """
        # KL divergence loss
        kl_div = self.kl_divergence(mu, logvar)

        # Reconstruction loss
        recon_loss = reconstruction_criterion(outputs, targets)

        # Combine losses with weights (optional)
        alpha = 1.0  # Weight for reconstruction loss
        beta = 0.1  # Weight for KL divergence loss (adjust as needed)
        combined_loss = alpha * recon_loss + beta * kl_div

        return combined_loss

    def sample(self, num_samples: int) -> torch.Tensor:
        z, _, _ = self.decode(torch.randn(num_samples, self.latent_dim))
        generate_sample, _, _ = self.forward(z)
        return generate_sample
