import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore")

import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class VAE(nn.Module):
    def __init__(self, input_dim: int = 1, latent_dim: int = 128, hidden_dim: List = None) -> None:
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dim is None:
            self.hidden_dim = [16, 32, 64, 128, 256]
        else:
            self.hidden_dim = hidden_dim

    def encoder(self) -> List[torch.Tensor]:
        in_channel = self.input_dim

        modules = []
        for dims in self.hidden_dim:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channel, dims, kernel_size=3, stride=2, padding=1)),
                nn.BatchNorm2d(dims),
            nn.PReLU())
            in_channel = dims

        self.encoder = nn.Sequential(*modules)
        
        mu = nn.Linear(self.hidden_dim[-1]*5, self.latent_dim)
        var = nn.Linear(self.hidden_dim[-1]*5, self.latent_dim)

        return [mu, var]
    def decoder(self):
