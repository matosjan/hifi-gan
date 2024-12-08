import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from torch import Tensor
from src.model.hifi_gan.resblock import ResBlock

class MRF(nn.Module):
    def __init__(self, n_channels, k_r, D_r):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResBlock(n_channels=n_channels, kernel_size=k_r[i], dilations=D_r[i]) for i in range(len(k_r))])

    def forward(self, x: Tensor):
        x = self.res_blocks[0](x)
        for i in range(1, len(self.res_blocks)):
            x = self.res_blocks[i](x)

        return x