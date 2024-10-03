from einops import rearrange
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
# from .bdl_losses import DiceLoss

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None, b=None):
        if mask is not None:
            logits_r = rearrange(logits, '(b l) c h w -> b l c h w', b=b)
            logits_e = torch.cat([logits_r[:, 1:], logits_r[:, -1:]], dim=1)
            loss_mse = torch.nn.SmoothL1Loss(reduction='mean')(logits_e, logits_r)
            # loss_mse = loss_mse * rearrange(mask, '(b l)-> b l', b=b)
            # print(loss_mse.shape)
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        if mask is not None:
            return 1 - dice.mean() + loss_mse * 0.1
        else:
            return 1 - dice.mean()


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
    # target = rearrange(target, '(b l) h w -> b l h w', b=b)
    
    mask = torch.zeros(b,l).to(target.device)
    mask[:,-1]=1
    mask = rearrange(mask, 'b l -> (b l)')

    # criterion_res = MultiClassSegmentationLoss(num_classes=3).cuda()
    criterion_res = mask_DiceLoss(3)
    criterion_con = nn.SmoothL1Loss()

    output_target = outputs['seg_final']
    out1 = outputs['out1']
    out2 = outputs['out2']
    out3 = outputs['out3']
    
    target_4d = target.float().unsqueeze(1)
    # target0 = F.interpolate(target_4d, scale_factor=0.5, mode='bilinear', align_corners=True)
    target0 = F.interpolate(target_4d, size=(368, 640), mode='bilinear')
    loss0 = criterion_res(output_target, target0, mask, b)
    
    target1 = F.interpolate(target0, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion_res(out1, target1, mask, b)
    
    target2 = F.interpolate(target1, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion_res(out2, target2, mask, b)
    
    target3 = F.interpolate(target2, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion_res(out3, target3, mask, b)
    
    loss_ce = (loss0 + loss1 + loss2 + loss3 ) / 4
    
    output_target = rearrange(output_target, '(b l) c h w -> b l c h w', b=b)
    out1 = rearrange(out1, '(b l) c h w -> b l c h w', b=b)
    out2 = rearrange(out2, '(b l) c h w -> b l c h w', b=b)
    out3 = rearrange(out3, '(b l) c h w -> b l c h w', b=b)

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