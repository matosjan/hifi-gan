import torch
import torch.nn as nn
from torch import Tensor

from src.model.hifi_gan.generator import Generator
from src.model.hifi_gan.mpd import MPD
from src.model.hifi_gan.msd import MSD


class HiFiGAN(nn.Module):
    def __init__(self, generator_config, mpd_periods):
        super().__init__()
        self.generator = Generator(**generator_config)
        self.mpd = MPD(mpd_periods)
        self.msd = MSD()

    def forward(self, melspec: Tensor, **batch):
        return self.generator(melspec)
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
