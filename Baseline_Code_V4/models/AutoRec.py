import torch.nn as nn
import numpy as np


class AutoRec(nn.Module):
    def __init__(self, margs):
        super(AutoRec, self).__init__()
        self.input_dim = margs.data_instance.num_item
        self.hidden_dim = margs.hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.input_dim),
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
