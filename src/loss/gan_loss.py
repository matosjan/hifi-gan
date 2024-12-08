import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor

from src.loss.adv_loss import AdvDiscriminatorLoss, AdvGeneratorLoss
from src.loss.mel_loss import MelSpectrogramLoss
from src.loss.fm_loss import FeatureMatchingLoss

class GeneratorLoss(nn.Module):
    def __init__(self, lambda_fm, lambda_mel) -> None:
        super().__init__()
        
        self.gen_adv_loss = AdvGeneratorLoss()
        self.fm_loss = FeatureMatchingLoss(lambda_fm)
        self.mel_loss = MelSpectrogramLoss(lambda_mel)

    def forward(self, **batch):
        fm_loss = self.fm_loss(**batch)
        mel_loss = self.mel_loss(**batch)
        adv_gen_loss = self.gen_adv_loss(**batch)
        return {'gen_loss': adv_gen_loss + fm_loss + mel_loss,
                'adv_gen_loss': adv_gen_loss, 
                'fm_loss': fm_loss, 
                'mel_loss': mel_loss}

class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_fm, lambda_mel):
        super().__init__()

        self.gen_loss = GeneratorLoss(lambda_fm=lambda_fm, lambda_mel=lambda_mel)
        self.disc_loss = AdvDiscriminatorLoss()

        