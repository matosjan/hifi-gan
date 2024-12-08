import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

class MelSpectrogramLoss(nn.Module):
    def __init__(self, lambda_mel) -> None:
        super().__init__()
        self.lambda_mel = lambda_mel

    def forward(self, pred_melspec: Tensor, melspec: Tensor, **batch) -> Tensor:
        return self.lambda_mel * F.l1_loss(pred_melspec, melspec, reduction='mean') 
