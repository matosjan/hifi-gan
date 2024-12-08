import torch
from torch import Tensor
from torch import nn

class AdvDiscriminatorLoss(nn.Module):
    def forward(self, mpd_pred_on_pred_list: Tensor, mpd_pred_on_gt_list: Tensor, msd_pred_on_pred_list: Tensor, msd_pred_on_gt_list: Tensor, **batch) -> Tensor:
        mpd_loss = 0.0
        msd_loss = 0.0
        for mpd_pred_on_pred, mpd_pred_on_gt, msd_pred_on_pred, msd_pred_on_gt in zip(mpd_pred_on_pred_list, mpd_pred_on_gt_list, msd_pred_on_pred_list, msd_pred_on_gt_list):
            mpd_loss += torch.mean((mpd_pred_on_gt - 1) ** 2) + torch.mean(mpd_pred_on_pred ** 2)
            msd_loss += torch.mean((msd_pred_on_gt - 1) ** 2) + torch.mean(msd_pred_on_pred ** 2)

        return {'disc_loss': mpd_loss + msd_loss, 
                'mpd_loss': mpd_loss, 
                'msd_loss': msd_loss}
    
class AdvGeneratorLoss(nn.Module):
    def forward(self, mpd_pred_on_pred_list: Tensor, msd_pred_on_pred_list, **batch) -> Tensor:
        mpd_loss = 0.0
        msd_loss = 0.0
        for mpd_pred_on_pred, msd_pred_on_pred in zip(mpd_pred_on_pred_list, msd_pred_on_pred_list):
            mpd_loss += torch.mean((mpd_pred_on_pred - 1)**2)
            msd_loss += torch.mean((msd_pred_on_pred - 1)**2)
        return mpd_loss + msd_loss
    

# import torch
# from torch import nn
# import torch.nn.functional as F


# class DiscriminatorAdvLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, gt_outputs, pred_outputs):
#         total_loss = 0.0
#         for gt_output, pred_output in zip(gt_outputs, pred_outputs):
#             gt_loss = torch.mean((gt_output - 1) ** 2)
#             pred_loss = torch.mean(pred_output ** 2)
#             total_loss += gt_loss + pred_loss
#         return total_loss

        
# class GeneratorAdvLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, pred_outputs):
#         total_loss = 0.0
#         for pred_output in pred_outputs:
#             pred_loss = torch.mean((pred_output - 1) ** 2)
#             total_loss += pred_loss
#         return total_loss