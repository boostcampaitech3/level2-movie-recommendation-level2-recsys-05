import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, margs):
        super(MultiDAE, self).__init__()
        self.p_dims = margs.p_dims + [margs.data_instance.num_item]
        self.q_dims = self.p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
            ]
        )
        self.drop = nn.Dropout(margs.dropout_rate)

        self.init_weights()

    def forward(self, input, calculate_loss=True):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)

        if calculate_loss:
            BCE_loss = -torch.mean(torch.sum(F.log_softmax(h, 1) * input, -1))
            return BCE_loss

        else:
            return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
