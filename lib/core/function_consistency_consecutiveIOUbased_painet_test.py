# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import os
import logging
# import os
import time

import numpy as np
# import numpy.ma as ma
from tqdm import tqdm

import torch
# import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

# from ..utils.utils import AverageMeter
from ..utils.utils import get_confusion_matrix
# from ..utils.utils import adjust_learning_rate
from ..utils.utils import get_world_size, get_rank
# import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw
import cv2
from ..co_tracker.fun_pseudolabel_generator import predict_propagation
from einops import rearrange

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def validate(config, testloader, model, device):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    total_num_inThresh = np.zeros(4)
    total_num_points = 0
    total_distance = 0
    total_num_Present= 0 

    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            images, labels, cpts_gt, cpts_presence, _ = batch
            images = rearrange(images, 'b l c h w -> (b l) c h w')
            labels = rearrange(labels, 'b l h w -> (b l) h w')
            cpts_gt = rearrange(cpts_gt, 'b l x y -> (b l) x y')
            cpts_presence = rearrange(cpts_presence, 'b l x y -> (b l) x y')
            size = labels.size()
            images = images.to(device)
            labels = labels.long().to(device)
            cpts_gt = cpts_gt.to(device)
            cpts_presence = cpts_presence.to(device)
            
            seg_out, cpts_out = model(images)
            seg_out = F.interpolate(input=seg_out, size=(size[-2], size[-1]), mode='bilinear')            
            cpts_out = torch.reshape(cpts_out, (cpts_out.size(0), -1, 2))
            
            #==================将PAINET 的结果调整成和我们的结果格式一致=====start===============
            # PAINET 的预测结果中，landmark 和 segmentation 都输出了六类的结果 class 0 -'left carotid'， 1- 'left optic protuberance', 2 - 'right carotid', 3 - "right optic protuberance'， 4 - 'clival recess', 5 - 'sella'
            # 我的dataset 中，landmark: class 0 -'left carotid'， 1- 'left optic protuberance', 2 - "right optic protuberance', 3 - 'right carotid';
            # segmentation: 0 - backgaound 1 - 'sella'; 2- 'clival recess'
            # landmark 步骤1: 删除最后两个点的预测值
            cpts_out = cpts_out[:, :4, :]
            # 步骤2: 交换第三个和第四个坐标的顺序
            cpts_out[:, [2, 3]] = cpts_out[:, [3, 2]]
            
            # segmentation 
            # 步骤1: 创建一个新的张量，包含背景类（索引0）
            batch_size, _, height, width = seg_out.shape
            adjusted = torch.zeros((batch_size, 3, height, width), device=seg_out.device)
            # 步骤2: 填充新张量
            # 背景类（索引0）保持为0
            # 原class 4（索引5）-> 新class 1
            # 原class 5（索引4）-> 新class 2
            adjusted[:, 1] = seg_out[:, 5]
            adjusted[:, 2] = seg_out[:, 4]
            # 步骤3: 计算背景类（所有其他类别的反面）
            adjusted[:, 0] = -torch.max(seg_out, dim=1)[0]
            seg_out = adjusted
            #==================将PAINET 的结果调整成和我们的结果格式一致======end==============            
            
            # calculate euclidean_distance between predicted and ground-truth landmarks
            labels = rearrange(labels, '(b l) h w -> b l h w', b=config.TEST.BATCH_SIZE_PER_GPU)[:,-1]
            seg_out = rearrange(seg_out, '(b l) c h w -> b l c h w', b=config.TEST.BATCH_SIZE_PER_GPU)[:,-1]
            cpts_presence = rearrange(cpts_presence, '(b l) x y -> b l x y', b=config.TEST.BATCH_SIZE_PER_GPU)[:,-1]
            cpts_out      = rearrange(cpts_out, '(b l) x y -> b l x y', b=config.TEST.BATCH_SIZE_PER_GPU)[:,-1]
            cpts_gt       = rearrange(cpts_gt, '(b l) x y -> b l x y', b=config.TEST.BATCH_SIZE_PER_GPU)[:,-1]
            squared_distance = torch.square(cpts_out - cpts_gt).detach() * cpts_presence
            squared_distance[:, :, 0] *= 1280**2
            squared_distance[:, :, 1] *= 720**2
            euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=2))
            thresholds = [36, 72, 108, 144]
            for i, thresh in enumerate(thresholds):
                num_inThresh = ((euclidean_distance <= thresh) * cpts_presence[:, :, 0]).float()
                total_num_inThresh[i] += torch.sum(num_inThresh).item() 
            
            total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)
            total_num_Present += torch.sum(cpts_presence[:, :, 0])
            total_distance += torch.sum(euclidean_distance)
    
            confusion_matrix += get_confusion_matrix(
                labels,
                seg_out,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    accuracy = tp.sum()/pos.sum()*100
    recall = (tp/np.maximum(1.0, pos))*100
    precision = (tp/np.maximum(1.0, res))*100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))*100
    mean_IoU = IoU_array[-2:].mean()
    
    mean_distance = total_distance/total_num_Present
    mpck = total_num_inThresh/total_num_Present.cpu().numpy()*100
    
    logging.info('Val_Metric==> IoU_array:{}, mean_IoU:{}, Recall:{}, Accuracy:{}, Precision:{}, MPCK:{}, mean_Distance:{: 4.4f}'.format(
                IoU_array, mean_IoU, recall, accuracy, precision, mpck, mean_distance))


def test(config, testloader, model, sv_dir='', sv_pred=True, device=None, temp_length=3):
    model.eval()
    total_num_points = 0
    total_num_inThresh = np.zeros(4)
    total_distance = 0
    total_num_Present= 0 
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
        
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            images, labels, _, _, name = batch
            name = [item for sublist in name for item in sublist]

            cpts_presence = torch.from_numpy(np.ones([2, 4, 2]))
            size = labels.size()
            images = images.to(device)
            cpts_presence = cpts_presence.to(device)
            total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)

            seg_out, cpts_out = model(images.view(-1, 3, 736, 1280) )
            seg_out = F.interpolate(input=seg_out, size=(size[-2], size[-1]), mode='bilinear')
            cpts_out = torch.reshape(cpts_out, (cpts_out.size(0), -1, 2))
            
            #==================将PAINET 的结果调整成和我们的结果格式一致=====start===============
            # PAINET 的预测结果中，landmark 和 segmentation 都输出了六类的结果 class 0 -'left carotid'， 1- 'left optic protuberance', 2 - 'right carotid', 3 - "right optic protuberance'， 4 - 'clival recess', 5 - 'sella'
            # 我的dataset 中，landmark: class 0 -'left carotid'， 1- 'left optic protuberance', 2 - "right optic protuberance', 3 - 'right carotid';
            # segmentation: 0 - backgaound 1 - 'sella'; 2- 'clival recess'
            # landmark 步骤1: 删除最后两个点的预测值
            cpts_out = cpts_out[:, :4, :]
            # 步骤2: 交换第三个和第四个坐标的顺序
            cpts_out[:, [2, 3]] = cpts_out[:, [3, 2]]
            
            # segmentation 
            # 步骤1: 创建一个新的张量，包含背景类（索引0）
            batch_size, _, height, width = seg_out.shape
            adjusted = torch.zeros((batch_size, 3, height, width), device=seg_out.device)
            # 步骤2: 填充新张量
            # 背景类（索引0）保持为0
            # 原class 4（索引5）-> 新class 1
            # 原class 5（索引4）-> 新class 2
            adjusted[:, 1] = seg_out[:, 5]
            adjusted[:, 2] = seg_out[:, 4]
            # 步骤3: 计算背景类（所有其他类别的反面）
            adjusted[:, 0] = -torch.max(seg_out, dim=1)[0]
            seg_out = adjusted
            #==================将PAINET 的结果调整成和我们的结果格式一致======end==============
            
            # here we replace the ground truth with the tracked predicted one for consistency calculation
            labels = []
            cpts_gt = []
            for i in range(temp_length-1):
                pred_mask, pred_cpts = predict_propagation(config, name, seg_out[i], cpts_out[i], cpts_presence[i], i)
                labels.append(torch.from_numpy(pred_mask))
                cpts_gt.append(torch.from_numpy(pred_cpts))
            labels = torch.stack(labels, dim=0).to(device)
            cpts_gt = torch.stack(cpts_gt, dim=0).to(device)
            images = images.squeeze(0)[1:]
            images = F.interpolate(input=images, size=(
                size[-2], size[-1]), mode='bilinear')
            seg_out = seg_out[1:]
            cpts_out = cpts_out[1:]
            cpts_presence = (cpts_gt!=-100).to(torch.int32)
            name = name[-temp_length:][1:]
                       
            # calculate euclidean_distance between predicted and ground-truth landmarks
            squared_distance = torch.square(cpts_out - cpts_gt).detach() * cpts_presence
            squared_distance[:, :, 0] *= 1280**2
            squared_distance[:, :, 1] *= 720**2
            euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=2))
            thresholds = [36, 72, 108, 144]
            for i, thresh in enumerate(thresholds):
                num_inThresh = ((euclidean_distance <= thresh) * cpts_presence[:, :, 0]).float()
                total_num_inThresh[i] += torch.sum(num_inThresh).item()  
            
            total_num_Present += torch.sum(cpts_presence[:, :, 0])
            total_distance += torch.sum(euclidean_distance)
            
            confusion_matrix += get_confusion_matrix(
                labels,
                seg_out,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            
            if sv_pred:
                sv_path = sv_dir
                # sv_path = os.path.join(sv_dir, 'PitVideo_Pred_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                seg_out = seg_out.cpu().numpy().copy()
                images = images.cpu().numpy().copy()
                ground_truth_landmarks = cpts_gt.cpu().numpy().copy()
                predicted_landmarks = cpts_out.cpu().numpy().copy()
                labels = labels.cpu().numpy().copy()
                seg_out = np.asarray(np.argmax(seg_out, axis=1), dtype=np.uint8)
                for i in range(seg_out.shape[0]):
                    ori_image = (
                        (images[i].transpose(1, 2, 0) * std) + mean)*255
                    result_image = ori_image.copy()
                    # Define colors
                    transparent = (0, 0, 0, 0)  # Transparent
                    cls_colors = [(77, 77, 255, 200), (255, 255, 77, 200), 
                                  (180, 77, 224, 255), (77, 255, 77, 255),
                                  (122, 233, 222, 255), (255, 77, 255, 255)]
                    
                    # Overlay ground truth mask1_contour
                    ground_truth_mask = np.array(labels[i], dtype=np.uint8)
                    mask = (ground_truth_mask == 1).astype(np.uint8)
                    _, binary= cv2.threshold(mask, 0.5, 255, 0)
                    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(result_image, contours, -1, (77, 77, 255),  3) 
                    # Overlay ground truth mask2_contour
                    mask = (ground_truth_mask == 2).astype(np.uint8)
                    _, binary= cv2.threshold(mask, 0.5, 255, 0)
                    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(result_image, contours, -1, (255, 255, 77),  3) 
                    
                    # conver from numpy to PIL
                    result_image = Image.fromarray(np.uint8(result_image))
                    draw = ImageDraw.Draw(result_image) 
                    
                    # # Overlay ground truth mask1 and 2, comment this section if only want to plot contour
                    # ground_truth_mask = np.array(labels[i], dtype=np.uint8)
                    # mask_image = Image.new(
                    #     "RGBA", result_image.size, transparent)
                    # for class_id in [1, 2]:
                    #     mask = (ground_truth_mask == class_id).astype(
                    #         np.uint8) * 200
                    #     mask_image.paste(
                    #         cls_colors[class_id-1], (0, 0), Image.fromarray(mask).convert("L"))
                    # result_image.paste(mask_image, (0, 0), mask_image)
                                        
                    
                    # Overlay predicted mask1             
                    predicted_mask = np.array(seg_out[i], dtype=np.uint8)
                    cls1_mask = Image.new("RGBA", result_image.size, transparent)
                    mask = (predicted_mask == 1).astype(np.uint8) * 200
                    cls1_mask.paste(cls_colors[0], (0, 0), Image.fromarray(mask).convert("L")) 
                    result_image.paste(cls1_mask, (0, 0), cls1_mask)
                    # Overlay predicted mask2
                    cls2_mask = Image.new("RGBA", result_image.size, transparent)
                    mask = (predicted_mask == 2).astype(np.uint8) * 200
                    cls2_mask.paste(cls_colors[1], (0, 0), Image.fromarray(mask).convert("L")) 
                    result_image.paste(cls2_mask, (0, 0), cls2_mask)

                    # Overlay ground truth landmarks as circles
                    j = 2
                    for landmark in ground_truth_landmarks[i]:
                        x, y = landmark
                        if x != 0 and y != 0:
                            draw.ellipse(
                                [(x*size[-1] - 20, y*size[-2] - 20), (x*size[-1] + 20, y*size[-2] + 20)], fill=cls_colors[j])
                        j += 1
                    
                    # # Overlay predicted landmarks as crosses
                    j =2
                    for landmark in predicted_landmarks[i]:
                        x, y = landmark
                        if x != 0 and y != 0:
                            draw.line([(x*size[-1] - 15, y*size[-2] - 15), (x*size[-1] + 15,
                                      y*size[-2] + 15)], fill=cls_colors[j], width=6)
                            draw.line([(x*size[-1] + 15, y*size[-2] - 15), (x*size[-1] - 15,
                                      y*size[-2] + 15)], fill=cls_colors[j], width=6)
                        j += 1
                    result_image.save(os.path.join(sv_path, name[i]))
                    
        confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)

        confusion_matrix = reduced_confusion_matrix.cpu().numpy()
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        accuracy = tp.sum()/pos.sum()*100
        recall = (tp/np.maximum(1.0, pos))*100
        precision = (tp/np.maximum(1.0, res))*100
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))*100
        mean_IoU = IoU_array[-2:].mean()
        
        mean_distance = total_distance/total_num_Present
        mpck = total_num_inThresh/total_num_Present.cpu().numpy()*100
        logging.info('Test_Metric==> IoU_array:{}, mean_IoU:{}, Recall:{}, Accuracy:{}, Precision:{}, MPCK:{}, mean_Distance:{: 4.4f}'.format(
                IoU_array, mean_IoU, recall, accuracy, precision, mpck, mean_distance))

