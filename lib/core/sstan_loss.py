from einops import rearrange
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .bdl_losses import DiceLoss


class MultiClassSegmentationLoss(nn.Module):
    def __init__(self, num_classes, weight_factor=5):
        super(MultiClassSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_factor = weight_factor

    def forward(self, pred, mask):
        # Ensure pred is [B, C, H, W] and mask is [B, H, W]
        assert pred.dim() == 4, "Prediction should be 4-dimensional"
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1).long()
        
        
        # Compute CE Loss
        ce_loss = F.cross_entropy(pred, mask, reduction='none')
        
        # Compute weights
        kernel_size = 31
        avg_mask = F.avg_pool2d(
            F.one_hot(mask, num_classes=self.num_classes).permute(0, 3, 1, 2).float(), 
            kernel_size=kernel_size, stride=1, padding=kernel_size//2
        )
        weights = 1 + self.weight_factor * torch.abs(avg_mask - F.one_hot(mask, num_classes=self.num_classes).permute(0, 3, 1, 2).float())
        
        # Weighted CE Loss
        wce_loss = (weights * ce_loss.unsqueeze(1)).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))
        wce_loss = wce_loss.mean()

        # Compute Dice Loss
        pred_softmax = F.softmax(pred, dim=1)
        mask_onehot = F.one_hot(mask, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred_softmax * mask_onehot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + mask_onehot.sum(dim=(2, 3))
        dice_score = (2. * intersection + 1) / (union + 1)
        dice_loss = 1 - dice_score.mean()

        # Combine losses
        total_loss = wce_loss + dice_loss
        return total_loss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, pred, mask):
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(dim=1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        smooth = 1
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        mask_flat = mask.view(size, -1)
        intersection = pred_flat * mask_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + mask_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return (wbce + dice_loss).mean()

def lossStage1(outputs, target, b):
    l = target.size(0)//b
    target = rearrange(target, '(b l) h w -> b l h w', b=b)

    # criterion_res = MultiClassSegmentationLoss(num_classes=3).cuda()
    criterion_res = DiceLoss(idc=[0, 1, 2])
    criterion_con = nn.SmoothL1Loss()

    output_target = outputs['seg_final']
    out1 = outputs['out1']
    out2 = outputs['out2']
    out3 = outputs['out3']
    
    target_4d = target[:,-1].float().unsqueeze(1)
    # target0 = F.interpolate(target_4d, scale_factor=0.5, mode='bilinear', align_corners=True)
    target0 = F.interpolate(target_4d, size=(368, 640), mode='bilinear')
    loss0 = criterion_res(output_target[:,-1], target0)
    
    target1 = F.interpolate(target0, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion_res(out1[:,-1], target1)
    
    target2 = F.interpolate(target1, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion_res(out2[:,-1], target2)
    
    target3 = F.interpolate(target2, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion_res(out3[:,-1], target3)
    
    loss_ce = (loss0 + loss1 + loss2 + loss3 ) / 4
    
    # output_target = rearrange(output_target, '(b l) c h w -> b l c h w', b=b)
    # out1 = rearrange(out1, '(b l) c h w -> b l c h w', b=b)
    # out2 = rearrange(out2, '(b l) c h w -> b l c h w', b=b)
    # out3 = rearrange(out3, '(b l) c h w -> b l c h w', b=b)

    loss_mse = 0
    for j in range(l - 1):
        loss_mse += criterion_con(output_target[:, j], output_target[:, j + 1])
        loss_mse += criterion_con(out1[:, j], out1[:, j + 1])
        loss_mse += criterion_con(out2[:, j], out2[:, j + 1])
        loss_mse += criterion_con(out3[:, j], out3[:, j + 1])
    loss_mse /= ((l - 1) * 4)

    weight = 0.1
    loss = (1 - weight) * loss_ce + weight * loss_mse
    # print(loss_ce, loss_mse, loss)

    return loss