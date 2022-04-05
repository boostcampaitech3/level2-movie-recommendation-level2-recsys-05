import torch.nn as nn
import numpy as np


class AutoRec(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoRec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.mse = nn.MSELoss()

        self.init_weights()

    def forward(self, input, calculate_loss=True):
        latent = self.encoder(input)
        recon_mat = self.decoder(latent)
        if calculate_loss:
            mse_loss = self.mse(recon_mat, input)
            return mse_loss

        else:
            return recon_mat

    def init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)

        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
