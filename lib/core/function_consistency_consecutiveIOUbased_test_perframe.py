# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import os
import logging
# import os
import time
import pandas as pd
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

def get_iou(pred_mask, true_mask, class_id):
    pred = (pred_mask == class_id).astype(int)
    true = (true_mask == class_id).astype(int)
    intersection = np.logical_and(pred, true)
    union = np.logical_or(pred, true)
    return np.sum(intersection) / np.sum(union)

def get_precision(pred_mask, true_mask, class_id):
    pred = (pred_mask == class_id).astype(int)
    true = (true_mask == class_id).astype(int)
    tp = np.sum(np.logical_and(pred, true))
    fp = np.sum(pred) - tp
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def get_recall(pred_mask, true_mask, class_id):
    pred = (pred_mask == class_id).astype(int)
    true = (true_mask == class_id).astype(int)
    tp = np.sum(np.logical_and(pred, true))
    fn = np.sum(true) - tp
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def get_f1_score(pred_mask, true_mask, class_id):
    precision = get_precision(pred_mask, true_mask, class_id)
    recall = get_recall(pred_mask, true_mask, class_id)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def test(config, testloader, model, sv_dir='', sv_pred=True, device=None, temp_length=3):
  model.eval()
  total_num_points = 0
  data = []
  
  with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            images, labels, cpts_gt, cpts_presence, name = batch
            name = [item for sublist in name for item in sublist]

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
            
            # Calculate per-frame metrics against ground truth labels
            row = [name[-1]]
            labels_last = rearrange(labels, '(b l) h w -> b l h w', b=1)[:,-1]
            seg_out_last = rearrange(seg_out, '(b l) c h w -> b l c h w', b=1)[:,-1]
            
            for class_id in range(1,3):
                iou = get_iou(torch.argmax(seg_out_last, dim=1).cpu().numpy(), labels_last.cpu().numpy(), class_id)
                row.append(iou)
            
            for class_id in range(1,3):
                f1_score = get_f1_score(torch.argmax(seg_out_last, dim=1).cpu().numpy(), labels_last.cpu().numpy(), class_id)
                row.append(f1_score)
            
            cpts_presence_last = rearrange(cpts_presence, '(b l) x y -> b l x y', b=1)[:,-1]
            cpts_out_last      = rearrange(cpts_out, '(b l) x y -> b l x y', b=1)[:,-1]
            cpts_gt_last       = rearrange(cpts_gt, '(b l) x y -> b l x y', b=1)[:,-1]
            squared_distance = torch.square(cpts_out_last - cpts_gt_last).detach() * cpts_presence_last
            squared_distance[:, :, 0] *= 1280**2
            squared_distance[:, :, 1] *= 720**2
            euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=2))
            
            thresholds = [36, 72, 108, 144]
            total_num_inThresh1 = np.zeros(4)
            for i, thresh in enumerate(thresholds):
                num_inThresh = ((euclidean_distance <= thresh) * cpts_presence_last[:, :, 0]).float()
                total_num_inThresh1[i] += torch.sum(num_inThresh).item() 
            
            mean_euclidean_distance = torch.mean(euclidean_distance, dim=1)
            if torch.sum(cpts_presence_last[:, :, 0]) > 0:
                mpck = total_num_inThresh1 / torch.sum(cpts_presence_last[:, :, 0]).cpu().numpy()
            else:
                mpck = -np.ones(4)
            
            row.extend(mpck.tolist())
            row.extend(mean_euclidean_distance.cpu().numpy().tolist())
                        
            # calculate consistency metrics
            # here we replace the ground truth with the tracked predicted one for consistency calculation
            labels = []
            cpts_gt = []
            cpts_presence = torch.from_numpy(np.ones([2, 4, 2]))
            cpts_presence = cpts_presence.to(device)
            total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)
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
            
            for class_id in range(1,3):
              iou = get_iou(torch.argmax(seg_out[-1:], dim=1).cpu().numpy(), labels[-1:].cpu().numpy(), class_id)
              row.append(iou)
          
            for class_id in range(1,3):
              f1_score = get_f1_score(torch.argmax(seg_out[-1:], dim=1).cpu().numpy(), labels[-1:].cpu().numpy(), class_id)
              row.append(f1_score)
                        
            # calculate euclidean_distance between predicted and ground-truth landmarks
            squared_distance = torch.square(cpts_out[-1:] - cpts_gt[-1:]).detach() * cpts_presence[-1:]
            squared_distance[:, :, 0] *= 1280**2
            squared_distance[:, :, 1] *= 720**2
            euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=2))
            thresholds = [36, 72, 108, 144]
            total_num_inThresh2 = np.zeros(4)
            for i, thresh in enumerate(thresholds):
                num_inThresh = ((euclidean_distance <= thresh) * cpts_presence[-1:][:, :, 0]).float()
                total_num_inThresh2[i] += torch.sum(num_inThresh).item() 
            
            mean_euclidean_distance = torch.mean(euclidean_distance, dim=1)
            mpck = total_num_inThresh2 / torch.sum(cpts_presence[-1:][:, :, 0]).cpu().numpy()
            
            row.extend(mpck.tolist())
            row.extend(mean_euclidean_distance.cpu().numpy().tolist())
            
            data.append(row)
        
        # Create DataFrame
        columns = ['Name', 
                    'IoU_Class1', 'IoU_Class2', 
                    'F1_Class1', 'F1_Class2',
                    'MPCK_36', 'MPCK_72', 'MPCK_108', 'MPCK_144',
                    'Mean_Euclidean_Distance',
                    'Consistency_IoU_Class1', 'Consistency_IoU_Class2',
                    'Consistency_F1_Class1', 'Consistency_F1_Class2',
                    'Consistency_MPCK_36', 'Consistency_MPCK_72', 'Consistency_MPCK_108', 'Consistency_MPCK_144',
                    'Consistency_Mean_Euclidean_Distance']

        df = pd.DataFrame(data, columns=columns)
        
        # Save to Excel
        output_file = sv_dir
        df.to_excel(output_file, engine='openpyxl', index=False)
        print(f"Results saved to {output_file}")


# def test(config, testloader, model, sv_dir='', sv_pred=True, device=None, temp_length=3):
#     model.eval()
#     total_num_points = 0
#     header_df = pd.read_excel(sv_dir, nrows=0)
#     header = header_df.columns.tolist()
#     data = []
        
#     with torch.no_grad():
#         for _, batch in enumerate(tqdm(testloader)):
#             row = []
#             total_num_inThresh1 = np.zeros(4)
#             total_num_inThresh2 = np.zeros(4)
#             images, labels, cpts_gt, cpts_presence, name = batch
#             name = [item for sublist in name for item in sublist]

#             images = rearrange(images, 'b l c h w -> (b l) c h w')
#             labels = rearrange(labels, 'b l h w -> (b l) h w')
#             cpts_gt = rearrange(cpts_gt, 'b l x y -> (b l) x y')
#             cpts_presence = rearrange(cpts_presence, 'b l x y -> (b l) x y')
#             size = labels.size()
#             images = images.to(device)
#             labels = labels.long().to(device)
#             cpts_gt = cpts_gt.to(device)
#             cpts_presence = cpts_presence.to(device)
            
#             seg_out, cpts_out = model(images)
#             seg_out = F.interpolate(input=seg_out, size=(size[-2], size[-1]), mode='bilinear')
#             cpts_out = torch.reshape(cpts_out, (cpts_out.size(0), -1, 2))
            
#             # calculate perframe metrics against groud truth labels
#             row.append(name[-1])
#             labels_last = rearrange(labels, '(b l) h w -> b l h w', b=1)[:,-1]
#             seg_out_last = rearrange(seg_out, '(b l) c h w -> b l c h w', b=1)[:,-1]
#             for class_id in range(1,3):
#                 iou = get_iou(torch.argmax(seg_out_last, dim=1).cpu().numpy(), labels_last.cpu().numpy(), class_id)
#                 row.append(iou)
                
#             for class_id in range(1,3):
#                 f1_score = get_f1_score(torch.argmax(seg_out_last, dim=1).cpu().numpy(), labels_last.cpu().numpy(), class_id)
#                 row.append(f1_score)
            
#             cpts_presence_last = rearrange(cpts_presence, '(b l) x y -> b l x y', b=1)[:,-1]
#             cpts_out_last      = rearrange(cpts_out, '(b l) x y -> b l x y', b=1)[:,-1]
#             cpts_gt_last       = rearrange(cpts_gt, '(b l) x y -> b l x y', b=1)[:,-1]
#             squared_distance = torch.square(cpts_out_last - cpts_gt_last).detach() * cpts_presence_last
#             squared_distance[:, :, 0] *= 1280**2
#             squared_distance[:, :, 1] *= 720**2
#             euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=2))
#             thresholds = [36, 72, 108, 144]
#             for i, thresh in enumerate(thresholds):
#                 num_inThresh = ((euclidean_distance <= thresh) * cpts_presence_last[:, :, 0]).float()
#                 total_num_inThresh1[i] += torch.sum(num_inThresh).item() 
#             mean_euclidean_distance = torch.mean(euclidean_distance, dim=1)
#             if torch.sum(cpts_presence_last[:, :, 0])>0:
#                 mpck = total_num_inThresh1/torch.sum(cpts_presence_last[:, :, 0]).cpu().numpy()
#             else:
#                 mpck = -np.ones(4)
#             row.append(mpck)
#             row.append(mean_euclidean_distance)
                        
#             # calculate consistency metrics
#             # here we replace the ground truth with the tracked predicted one for consistency calculation
#             labels = []
#             cpts_gt = []
#             cpts_presence = torch.from_numpy(np.ones([2, 4, 2]))
#             cpts_presence = cpts_presence.to(device)
#             total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)
#             for i in range(temp_length-1):
#                 pred_mask, pred_cpts = predict_propagation(config, name, seg_out[i], cpts_out[i], cpts_presence[i], i)
#                 labels.append(torch.from_numpy(pred_mask))
#                 cpts_gt.append(torch.from_numpy(pred_cpts))
#             labels = torch.stack(labels, dim=0).to(device)
#             cpts_gt = torch.stack(cpts_gt, dim=0).to(device)
#             images = images.squeeze(0)[1:]
#             images = F.interpolate(input=images, size=(
#                 size[-2], size[-1]), mode='bilinear')
#             seg_out = seg_out[1:]
#             cpts_out = cpts_out[1:]
#             cpts_presence = (cpts_gt!=-100).to(torch.int32)
#             name = name[-temp_length:][1:]
            
#             for class_id in range(1,3):
#                 iou = get_iou(torch.argmax(seg_out[-1], dim=1).cpu().numpy(), labels[-1].cpu().numpy(), class_id)
#                 row.append(iou)
#             for class_id in range(1,3):
#                 f1_score = get_f1_score(torch.argmax(seg_out[-1], dim=1).cpu().numpy(), labels[-1].cpu().numpy(), class_id)
#                 row.append(f1_score)
                       
#             # calculate euclidean_distance between predicted and ground-truth landmarks
#             squared_distance = torch.square(cpts_out[-1:] - cpts_gt[-1:]).detach() * cpts_presence[-1:]
#             squared_distance[:, :, 0] *= 1280**2
#             squared_distance[:, :, 1] *= 720**2
#             euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=2))
#             thresholds = [36, 72, 108, 144]
#             for i, thresh in enumerate(thresholds):
#                 num_inThresh = ((euclidean_distance <= thresh) * cpts_presence[-1:][:, :, 0]).float()
#                 total_num_inThresh2[i] += torch.sum(num_inThresh).item() 
                
#             mean_euclidean_distance = torch.mean(euclidean_distance, dim=1)
#             mpck = total_num_inThresh2/torch.sum(cpts_presence[-1:][:, :, 0]).cpu().numpy()
#             row.append(mpck)
#             row.append(mean_euclidean_distance) 
#             data.append(row)
            
#     # 创建DataFrame
#     df = pd.DataFrame(data, columns=header)
#     # 保存到CSV
#     df.to_excel(sv_dir, index=False)



