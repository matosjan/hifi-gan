import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import List


class FeatureMatchingLoss(nn.Module):
    def __init__(self, lambda_fm=1) -> None:
        super().__init__()
        self.lambda_fm = lambda_fm

    def forward(self, pred_fmap_lists: List[List[Tensor]], gt_fmap_lists: List[List[Tensor]], **batch) -> Tensor:
        total_loss = 0
        for pred_fmap_list_disc, gt_fmap_list_disc in zip(pred_fmap_lists, gt_fmap_lists):
            loss_for_disc = 0.0
            for pred_fmap_list, gt_fmap_list in zip(pred_fmap_list_disc, gt_fmap_list_disc):
                for pred_fmap, gt_fmap in zip(pred_fmap_list, gt_fmap_list):
                    loss_for_disc += F.l1_loss(pred_fmap, gt_fmap, reduction='mean')
                total_loss += self.lambda_fm * loss_for_disc
        return total_loss