import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
    
class MPDSubDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        
        self.conv_layers = nn.ModuleList([])

        out_channels_list = [32, 128, 512, 1024]
        for i, out_channels in enumerate(out_channels_list):
            in_channels = 1 if i == 0 else out_channels_list[i - 1]
            self.conv_layers.append(weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))))

        self.conv_layers.append(weight_norm(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(5, 1), padding='same')))
        self.end_conv = weight_norm(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(3, 1), padding='same'))

        self.activation = nn.LeakyReLU()

    def forward(self, x: Tensor):
        fmaps = []
        if x.shape[-1] % self.period > 0:
            x = F.pad(x, (0, self.period - x.shape[-1] % self.period), mode="reflect")
        x = x.reshape(x.shape[0], 1, x.shape[-1] // self.period, self.period)

        for conv in self.conv_layers:
            x = self.activation(conv(x))
            fmaps.append(x)
        x = self.end_conv(x)
        fmaps.append(x)

        return x.flatten(-2, -1), fmaps
        
    
class MPD(nn.Module):
    def __init__(self, periods):
        super().__init__()
        self.sub_mpd_blocks = nn.ModuleList([MPDSubDiscriminator(period=period) for period in periods])

    def forward(self, x: Tensor):
        preds = []
        fmaps = []
        for sub_block in self.sub_mpd_blocks:
            out, fmap = sub_block(x)
            preds.append(out)
            fmaps.append(fmap)

        return preds, fmaps

