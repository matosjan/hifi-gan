import torch
import torch.nn as nn
from typing import List
from torch import Tensor

from src.model.hifi_gan.mrf import MRF

class Generator(nn.Module):
    def __init__(self, n_mels, hidden_dim, k_u, k_r, D_r):
        super().__init__()

        self.first_conv = nn.utils.weight_norm(nn.Conv1d(in_channels=n_mels, out_channels=hidden_dim, 
                                                         kernel_size=7, dilation=1, padding="same"))

        self.mid_blocks = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.ConvTranspose1d(in_channels=(hidden_dim // 2**i), out_channels=(hidden_dim // 2**(i+1)), 
                                                        kernel_size=kernel_size, stride=kernel_size // 2, padding=(kernel_size - kernel_size // 2) // 2)),   
                MRF(n_channels=(hidden_dim // 2**(i+1)), k_r=k_r,D_r=D_r)
            )
            for i, kernel_size in enumerate(k_u)
        ])

        self.end_block = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_channels=(hidden_dim // (2**len(self.mid_blocks))), out_channels=1, kernel_size=7, padding='same')),
            nn.Tanh()
        )

        self.activation = nn.LeakyReLU()

    def forward(self, melspec: Tensor):
        x = self.first_conv(melspec)
        for block in self.mid_blocks:
            x = self.activation(x)
            x = block(x)

        x = self.activation(x)
        audio_pred = self.end_block(x)

        return {'audio_pred': audio_pred}
    
            






