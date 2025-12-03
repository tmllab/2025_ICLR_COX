import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VIB(nn.Module):

    def __init__(self, out_embed_dim=1024, hidden_dim=256):
        super(VIB, self).__init__()
        self.z_dim = hidden_dim

        self.encode = nn.Sequential(
            nn.Linear(out_embed_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2*self.z_dim))

        self.decode = nn.Sequential(
            nn.Linear(self.z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, out_embed_dim))


    def forward(self, x):
        if x.dim() > 2 :
            x = x.view(x.size(0),-1)

        statistics = self.encode(x)
        mu, std = statistics[:, :self.z_dim], statistics[:, self.z_dim:]

        encoding = self.reparameterize(mu, std)
        y = self.decode(encoding)

        return y

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    
    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()