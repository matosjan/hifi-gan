import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from torch import Tensor

class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, dilations):
        super().__init__()
        self.list_of_blocks = nn.ModuleList([])
        for m in range(len(dilations)):
            inner_blocks = []
            for l in range(len(dilations[m])):
                inner_blocks.append(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        weight_norm(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, 
                                              kernel_size=kernel_size, dilation=dilations[m][l], padding='same'))
                    )
                )
            self.list_of_blocks.append(nn.Sequential(*inner_blocks))
    
    def forward(self, x: Tensor):
        for inner_block in self.list_of_blocks:
            x = x + inner_block(x)
        
        return x