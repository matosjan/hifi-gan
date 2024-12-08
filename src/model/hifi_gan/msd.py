import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch import Tensor
from torch.nn.utils import weight_norm, spectral_norm

class MSDSubDiscriminator(nn.Module):
    def __init__(self, is_first):
        super().__init__()
        norm = None
        if is_first:
            norm = spectral_norm
        else:
            norm = weight_norm

        self.conv_layers = nn.ModuleList([])

        self.conv_layers.append(norm(nn.Conv1d(in_channels=1, out_channels=128, kernel_size=15, stride=1, groups=1, padding=7)))
        self.conv_layers.append(norm(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=41, stride=2, groups=4, padding=20)))
        self.conv_layers.append(norm(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=41, stride=2, groups=16, padding=20)))
        self.conv_layers.append(norm(nn.Conv1d(in_channels=256, out_channels=512, kernel_size=41, stride=4, groups=16, padding=20)))
        self.conv_layers.append(norm(nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=41, stride=4, groups=16, padding=20)))
        self.conv_layers.append(norm(nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=41, stride=1, groups=16, padding=20)))
        self.conv_layers.append(norm(nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, groups=1, padding=2)))
        
        self.last_conv = norm(nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1))
        self.activation = nn.LeakyReLU()

    def forward(self, x: Tensor):
        fmaps = []
        for conv in self.conv_layers:
            x = self.activation(conv(x))
            fmaps.append(x)
        x = self.last_conv(x)
        fmaps.append(x)

        return x, fmaps

class MSD(nn.Module):
    def __init__(self):
        super().__init__()

        self.poolings = [nn.Identity(), nn.AvgPool1d(kernel_size=4, stride=2, padding=2), nn.AvgPool1d(kernel_size=4, stride=2, padding=2)]
        self.sub_msd_blocks = nn.ModuleList([
            nn.Sequential(
                pooling_layer,
                MSDSubDiscriminator(is_first=True if i == 0 else False)
            )
            for i, pooling_layer in enumerate(self.poolings)
        ])


    def forward(self, x: Tensor):
        preds = []
        fmaps = []
        for sub_block in self.sub_msd_blocks:
            x, fmap = sub_block(x)
            preds.append(x)
            fmaps.append(fmap)

        return preds, fmaps