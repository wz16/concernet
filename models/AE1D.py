import torch
import torch.nn as nn
import importlib
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self, input_dim, latent_dim=9, **kwargs):
        super(Net, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, 16),
                torch.nn.Tanh(),
                torch.nn.Linear(16, latent_dim)
            )
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 16),
                torch.nn.Tanh(),
                torch.nn.Linear(16, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, input_dim)
            )
    def encode(self, x):
        z = self.encoder(x)
        return z
    def decode(self, z):
        h = self.decoder(z)
        return h
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

