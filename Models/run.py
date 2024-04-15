import glob
import random

import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from VAE import VAE
from vae_preprocessing import DataGenerator
from vae_trainer import VAETrainer

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = glob.glob("../Data/Extracted/*/*.jpg")
random.shuffle(data_path)
dataset = pd.DataFrame(data_path, columns=['image_path'])
train, test= train_test_split(dataset, test_size=0.25)

image_size = 128
batch_size = 8

train_data = DataGenerator(train, transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
]))

test_data = DataGenerator(test, transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
]))

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = VAE().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = model.loss
num_epochs = 50

trainer = VAETrainer(model, train_dataloader, test_dataloader, optim, criterion)
trainer.train(num_epochs)