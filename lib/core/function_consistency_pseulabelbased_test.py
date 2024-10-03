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

def test(config, testloader, model, sv_dir='', sv_pred=True, device=None, temp_length=3):
    model.eval()
    total_num_points = 0
    total_num_inThresh = 0
    total_distance = 0
    total_num_Present= 0 
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
        
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, label, cpts_gt, cpts_presence, name = batch
            name = [item for sublist in name for item in sublist]

            image, label, cpts_gt, cpts_presence = image.view(-1, 3, 736, 1280), label.view(-1, 720, 1280), cpts_gt.view(-1, 4, 2), cpts_presence.view(-1, 4, 2)
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)
            cpts_gt = cpts_gt.to(device)
            cpts_presence = cpts_presence.to(device)
            total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)

            seg_pre, cpts_pre = model(image)
            pred = F.interpolate(input=seg_pre, size=(
                size[-2], size[-1]), mode='bilinear')

            cpts_pre = torch.reshape(
                cpts_pre, (cpts_pre.size(0), -1, 2))
            
            # calculate euclidean_distance between predicted and ground-truth landmarks
            cpts_pre= cpts_pre*cpts_presence
            norm_squared_distance = torch.square(cpts_pre-cpts_gt) * cpts_presence
            squared_distance = torch.zeros_like(norm_squared_distance)
            squared_distance[:, :, 0] = norm_squared_distance[:, :, 0]*(size[-1]**2) 
            squared_distance[:, :, 1] = norm_squared_distance[:, :, 1]*(size[-2]**2)
            euclidean_distance = torch.sum(
                squared_distance, dim=(2), keepdim=True)
            euclidean_distance = torch.sqrt(euclidean_distance.squeeze(dim=2))
            # calculate how many points are within 144 pixels from their corresponding ground truth
            num_inThresh = ((euclidean_distance >= 0) & (
                euclidean_distance <= 36)).float()
            num_inThresh = num_inThresh*cpts_presence[:, :, 0]
            total_num_inThresh += torch.sum(num_inThresh)
            total_num_Present += torch.sum(cpts_presence[:, :, 0])
            total_distance += torch.sum(euclidean_distance)
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            
            if sv_pred:
                sv_path = sv_dir
                # sv_path = os.path.join(sv_dir, 'PitVideo_Pred_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                pred = pred.cpu().numpy().copy()
                image = image.cpu().numpy().copy()
                ground_truth_landmarks = cpts_gt.cpu().numpy().copy()
                predicted_landmarks = cpts_pre.cpu().numpy().copy()
                label = label.cpu().numpy().copy()
                pred = np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)
                for i in range(pred.shape[0]):
                    ori_image = (
                        (image[i].transpose(1, 2, 0) * std) + mean)*255
                    result_image = ori_image.copy()
                    # Define colors
                    transparent = (0, 0, 0, 0)  # Transparent
                    cls_colors = [(77, 77, 255, 200), (255, 255, 77, 200), 
                                  (180, 77, 224, 255), (77, 255, 77, 255),
                                  (122, 233, 222, 255), (255, 77, 255, 255)]
                    
                    # Overlay ground truth mask1_contour
                    ground_truth_mask = np.array(label[i], dtype=np.uint8)
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
                    # ground_truth_mask = np.array(label[i], dtype=np.uint8)
                    # mask_image = Image.new(
                    #     "RGBA", result_image.size, transparent)
                    # for class_id in [1, 2]:
                    #     mask = (ground_truth_mask == class_id).astype(
                    #         np.uint8) * 200
                    #     mask_image.paste(
                    #         cls_colors[class_id-1], (0, 0), Image.fromarray(mask).convert("L"))
                    # result_image.paste(mask_image, (0, 0), mask_image)
                                        
                    
                    # Overlay predicted mask1             
                    predicted_mask = np.array(pred[i], dtype=np.uint8)
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
        mpck10 = total_num_inThresh/total_num_Present*100
        logging.info('Test_Metric==> IoU_array:{}, mean_IoU:{}, Recall:{}, Accuracy:{}, Precision:{}, MPCK10:{: 4.4f}, mean_Distance:{: 4.4f}'.format(
                IoU_array, mean_IoU, recall, accuracy, precision, mpck10, mean_distance))


