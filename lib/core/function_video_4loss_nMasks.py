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

from ..utils.utils import AverageMeter
from ..utils.utils import get_confusion_matrix
# from ..utils.utils import adjust_learning_rate
from ..utils.utils import get_world_size, get_rank
# import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw
import cv2
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


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,trainloader, optimizer, model, Seg_loss, L1_smooth,
          Landmark_loss, Landmark_loss2, L1_smooth2, writer_dict, device, stage, loss_weight, scheduler=None):

    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_seg_loss = AverageMeter()
    # ave_bound_loss = AverageMeter()
    ave_Wing_loss = AverageMeter()
    ave_FL_loss = AverageMeter()
    
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    # global_steps is the number of epoches rather than steps
    global_steps = writer_dict['train_global_steps']
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    # total_num_inThresh1 = 0
    total_num_inThresh = np.zeros(4)  # 4 threshold [36, 72, 108, 144]
    total_num_points = 0
    total_distance = 0
    total_num_Present= 0 

    for i_iter, batch in enumerate(tqdm(trainloader)):
        images, labels, cpts_gt, cpts_presence, _= batch
        # images, labels, cpts_gt, cpts_presence = images.view(-1, 3, 736, 1280), labels.view(-1, 720, 1280), cpts_gt.view(-1, 4, 2), cpts_presence.view(-1, 4, 2)
        images = rearrange(images, 'b l c h w -> (b l) c h w')
        labels = rearrange(labels, 'b l h w -> (b l) h w')
        cpts_gt = rearrange(cpts_gt, 'b l x y -> (b l) x y')
        cpts_presence = rearrange(cpts_presence, 'b l x y -> (b l) x y')
        size = labels.size()
        images = images.to(device)
        labels = labels.long().to(device)
        cpts_gt = cpts_gt.to(device)
        cpts_presence = cpts_presence.to(device)
        total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)
        
        # dist_map_label = dist_map_label.to(device)

        seg_out, cpts_out = model(images)
        seg_out = F.interpolate(input=seg_out, size=(size[1], size[2]), mode='bilinear')
        
        # ph, pw = seg_out.size(2), seg_out.size(3)
        # h, w = labels.size(1), labels.size(2)
        # if ph != h or pw != w:
        #     seg_out = F.interpolate(input=seg_out, size=(h, w), mode='bilinear')

        seg_loss = Seg_loss(seg_out, labels)
        seg_out_t = rearrange(seg_out, '(b l) c h w -> b l c h w', b=config.TRAIN.BATCH_SIZE_PER_GPU)
        seg_out_t = torch.cat([seg_out_t[:,1:], seg_out_t[:,-1:]], dim=1)
        seg_out_t = rearrange(seg_out_t, 'b l c h w -> (b l) c h w')
        seg_loss2 = L1_smooth(seg_out, seg_out_t)
        
        cpts_out = torch.reshape(cpts_out, (cpts_gt.size(0), cpts_gt.size(1), cpts_gt.size(2)))
        if stage == 1:
            cpts_out = cpts_out.detach()
            # seg_loss2 = Seg_loss2(seg_out, dist_map_label)
            cpts_loss = torch.tensor(0.).to(device)
            cpts_loss2 = torch.tensor(0.).to(device)
            cpts_loss3 = torch.tensor(0.).to(device)
        else:  
            # seg_loss2 = Seg_loss2(seg_out, dist_map_label)
            cpts_loss = Landmark_loss(cpts_out, cpts_gt)
            cpts_loss2 = Landmark_loss2(cpts_out, cpts_gt)
            cpts_out_t = rearrange(cpts_out, '(b l) x y -> b l x y', b=config.TRAIN.BATCH_SIZE_PER_GPU)
            cpts_out_t = torch.cat([cpts_out_t[:,1:], cpts_out_t[:,-1:]], dim=1)
            cpts_out_t = rearrange(cpts_out_t, 'b l x y -> (b l) x y')
            cpts_loss3 = L1_smooth2(cpts_out, cpts_out_t)

        # # calculate euclidean_distance between predicted and ground-truth landmarks
        # norm_squared_distance = torch.square(cpts_out-cpts_gt).detach() * cpts_presence
        # squared_distance = torch.zeros_like(norm_squared_distance)
        # squared_distance[:, :, 0] = norm_squared_distance[:, :, 0]*(1280**2)
        # squared_distance[:, :, 1] = norm_squared_distance[:, :, 1]*(720**2)
        # euclidean_distance = torch.sum(
        #     squared_distance, dim=(2), keepdim=True)
        # euclidean_distance = torch.sqrt(euclidean_distance.squeeze(dim=2))
        # # calculate how many points are within 144 pixels from their corresponding ground truth
        # num_inThresh = ((euclidean_distance >= 0) & (
        #     euclidean_distance <= 144)).float()
        # num_inThresh = num_inThresh*cpts_presence[:, :, 0]
        # total_num_inThresh1 += torch.sum(num_inThresh)
        
        # 计算预测和真值标注点之间的欧氏距离
        squared_distance = torch.square(cpts_out - cpts_gt).detach() * cpts_presence
        squared_distance[:, :, 0] *= 1280**2
        squared_distance[:, :, 1] *= 720**2
        euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=2))
        # 定义多个阈值
        thresholds = [36, 72, 108, 144]
        # 计算每个阈值范围内的点的数量，并将其累加到 total_num_inThresh (NumPy数组)
        for i, thresh in enumerate(thresholds):
            num_inThresh = ((euclidean_distance <= thresh) * cpts_presence[:, :, 0]).float()
            total_num_inThresh[i] += torch.sum(num_inThresh).item()  # 转换为标量并累加到 NumPy 数组

        total_num_Present += torch.sum(cpts_presence[:, :, 0])
        total_distance += torch.sum(euclidean_distance)

        cpts_loss = cpts_loss * cpts_presence
        cpts_loss2 = cpts_loss2 * cpts_presence

        if torch.sum(cpts_presence) > 0:
            cpts_loss = torch.sum(cpts_loss) / torch.sum(cpts_presence)
            cpts_loss2 = torch.sum(cpts_loss2) / torch.sum(cpts_presence)
        else:
            cpts_loss = torch.sum(cpts_loss)
            cpts_loss2 = torch.sum(cpts_loss2)
            

        if torch.isnan(cpts_loss):
            print("cpts_loss is nan")
            
        if torch.isnan(cpts_loss2):
            print("cpts_loss2 is nan")
            
        if torch.isnan(cpts_loss3):
            print("cpts_loss3 is nan")

        if torch.isnan(seg_loss):
            print("seg_loss is nan")
            
        # if torch.isnan(seg_loss2):
        #     print("seg_loss2 is nan")
                        
        seg_loss      = seg_loss*loss_weight[0]
        seg_loss2     = seg_loss2*loss_weight[1]
        cpts_loss     = cpts_loss*loss_weight[2]
        cpts_loss2    = cpts_loss2*loss_weight[3]
        cpts_loss3    = cpts_loss3*loss_weight[4]
        loss          = seg_loss + seg_loss2 + cpts_loss + cpts_loss2 + cpts_loss3
        # loss          = seg_loss + cpts_loss + cpts_loss2

        
        if torch.isnan(loss):
            print("loss is nan")

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_seg_loss.update(seg_loss.item())
        # ave_bound_loss.update(seg_loss2.item())
        ave_Wing_loss.update(cpts_loss.item())
        ave_FL_loss.update(cpts_loss2.item())

        confusion_matrix += get_confusion_matrix(
            labels,
            seg_out,
            size,
            config.DATASET.NUM_CLASSES,
            config.TRAIN.IGNORE_LABEL)
        lr = base_lr * ((1-float(i_iter+cur_iters)/num_iters)**(0.9))
        
        if stage==1:
            optimizer.param_groups[1]['lr'] = lr
            optimizer.param_groups[2]['lr'] = lr
        else:
            optimizer.param_groups[1]['lr'] = lr
            optimizer.param_groups[2]['lr'] = lr
            optimizer.param_groups[3]['lr'] = lr
            # optimizer.param_groups[3]['lr'] = lr
            
    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    train_accuracy = tp.sum()/pos.sum()*100
    train_recall = (tp/np.maximum(1.0, pos))*100
    train_precision = (tp/np.maximum(1.0, res))*100
    train_IoU = (tp / np.maximum(1.0, pos + res - tp))*100
    train_mIoU = train_IoU[-2:].mean()
    train_mdistance = total_distance/total_num_Present
    train_mpck = total_num_inThresh/total_num_Present.cpu().numpy()*100
    train_total_loss = ave_loss.average()
    train_segmentation_loss = ave_seg_loss.average()
    # train_bound_loss = ave_bound_loss.average()
    train_Wing_loss = ave_Wing_loss.average()
    train_FL_loss = ave_FL_loss.average()
    

    msg = 'Epoch: [{}/{}], Time: {:.2f}, ' \
        'lr: {:.6f}, Train_total_Loss: {:.6f}, Train_seg_loss: {:.6f}, Train_Wing_loss: {:.6f}, Train_FL_loss: {:.6f}'.format(
            epoch, num_epoch, batch_time.average(
            ), optimizer.param_groups[1]['lr'], train_total_loss, train_segmentation_loss, train_Wing_loss, train_FL_loss
        )

    # msg = 'Epoch: [{}/{}], Time: {:.2f}, ' \
    #     'lr: {:.6f}, Train_total_Loss: {:.6f}, Train_seg_loss: {:.6f}, Train_bound_loss:{:.6f},Train_Wing_loss: {:.6f}, Train_FL_loss: {:.6f}'.format(
    #         epoch, num_epoch, batch_time.average(
    #         ), optimizer.param_groups[1]['lr'], train_total_loss, train_segmentation_loss, train_bound_loss, train_Wing_loss, train_FL_loss
    #     )
    # msg = 'Epoch: [{}/{}], Time: {:.2f}, ' \
    #     'lr: {:.6f}, Train_total_Loss: {:.6f}, Train_seg_loss: {:.6f},Train_cpts_loss: {:.6f}, Train_presence_loss: {:.6f}'.format(
    #         epoch, num_epoch, batch_time.average(
    #         ), lr, train_total_loss, train_segmentation_loss, train_centroid_loss, train_presence_loss
    #     )
    logging.info(msg)

    # Here we add_scalar every config.PRINT_FREQ.
    # Since in the same epoch, the global step is the same, when add_scalar, it will overwrite the previous one.
    writer.add_scalar('Loss/train_total_loss', train_total_loss, global_steps)
    writer.add_scalar('Seg_loss/train_segmentation_loss',
                      train_segmentation_loss, global_steps)
    # writer.add_scalar('Boundary_loss/train_boundary_loss',
    #                   train_bound_loss, global_steps)
    writer.add_scalar('Landmark_loss/train_landmark_loss',
                      train_Wing_loss, global_steps)
    writer.add_scalar('Landmark_loss2/train_landmark_loss2',
                      train_FL_loss, global_steps)
    writer.add_scalar('Mean_distance/train_mdistance',
                      train_mdistance, global_steps)
    writer.add_scalar('mIoU/train_mIoU', train_mIoU, global_steps)
    writer.add_scalar('MPCK15/train_mpck15', train_mpck[-2], global_steps)
    writer.add_scalar('Learning_rate/convlstm_seghead', optimizer.param_groups[1]['lr'], global_steps)
    writer.add_scalar('Learning_rate/landhead', optimizer.param_groups[3]['lr'], global_steps)

    writer_dict['train_global_steps'] = global_steps + 1
    return train_total_loss, train_mIoU, train_IoU, train_accuracy, train_recall, train_precision, train_mdistance, train_mpck


def validate(config, testloader, model, Seg_loss, Landmark_loss, Landmark_loss2, writer_dict, device, stage, loss_weight):
    model.eval()
    ave_loss = AverageMeter()
    ave_seg_loss = AverageMeter()
    # ave_bound_loss = AverageMeter()
    ave_Wing_loss = AverageMeter()
    ave_FL_loss = AverageMeter()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    # total_num_inThresh1 = 0
    total_num_inThresh = np.zeros(4)
    total_num_points = 0
    total_distance = 0
    total_num_Present= 0 
    # total_num_Absent=0
    # total_num_truePresent = 0
    # total_num_trueAbsent = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            # image, label, cpts_gt, cpts_presence, _, dist_map_label = batch
            # label, cpts_gt, cpts_presence, dist_map_label = label[:,-1,:,:], cpts_gt[:,-1,:,:], cpts_presence[:,-1,:,:], dist_map_label[:,-1,:,:,:]
            images, labels, cpts_gt, cpts_presence, _ = batch
            labels, cpts_gt, cpts_presence = labels[:,-1,:,:], cpts_gt[:,-1,:,:], cpts_presence[:,-1,:,:]
            size = labels.size()
            images = images.to(device)
            labels = labels.long().to(device)
            cpts_gt = cpts_gt.to(device)
            cpts_presence = cpts_presence.to(device)
            total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)
            
            # dist_map_label = dist_map_label.to(device)

            seg_out, cpts_out = model(images)
            seg_out = seg_out.unsqueeze(0)[:,-1,:,:,:] # here is a bug, if batch size is greater than 1
            cpts_out = cpts_out.unsqueeze(0)[:,-1]
            seg_out = F.interpolate(input=seg_out, size=(size[-2], size[-1]), mode='bilinear')
            seg_loss = Seg_loss(seg_out, labels)
            
            cpts_out = torch.reshape(cpts_out, (cpts_out.size(0), cpts_gt.size(1), cpts_gt.size(2)))
            if stage == 1:
                cpts_loss = torch.tensor(0.).to(device)
                cpts_loss2 = torch.tensor(0.).to(device)
            else:
                cpts_loss = Landmark_loss(cpts_out, cpts_gt)
                cpts_loss2 = Landmark_loss2(cpts_out, cpts_gt)
            
            # 计算预测和真值标注点之间的欧氏距离
            squared_distance = torch.square(cpts_out - cpts_gt).detach() * cpts_presence
            squared_distance[:, :, 0] *= 1280**2
            squared_distance[:, :, 1] *= 720**2
            euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=2))
            # 定义多个阈值
            thresholds = [36, 72, 108, 144]
            # 计算每个阈值范围内的点的数量，并将其累加到 total_num_inThresh (NumPy数组)
            for i, thresh in enumerate(thresholds):
                num_inThresh = ((euclidean_distance <= thresh) * cpts_presence[:, :, 0]).float()
                total_num_inThresh[i] += torch.sum(num_inThresh).item()  # 转换为标量并累加到 NumPy 数组
            
            
            total_num_Present += torch.sum(cpts_presence[:, :, 0])
            total_distance += torch.sum(euclidean_distance)
            
            # # calculate the number of presence
            # pre_presence = torch.where(torch.sigmoid(cpts_presence_pre).cpu() < torch.tensor(0.5), torch.tensor(0), torch.tensor(1)).to(device)
            # total_num_truePresent += torch.sum((pre_presence+cpts_presence[:, :, 0].to(torch.uint8)==2).long())
            # total_num_trueAbsent += torch.sum((pre_presence+cpts_presence[:, :, 0].to(torch.uint8)==0).long())

            cpts_loss = cpts_loss * cpts_presence
            cpts_loss2 = cpts_loss2 * cpts_presence
            if torch.sum(cpts_presence) > 0:
                cpts_loss = torch.sum(cpts_loss) / torch.sum(cpts_presence)
                cpts_loss2 = torch.sum(cpts_loss2) / torch.sum(cpts_presence)
            else:
                cpts_loss = torch.sum(cpts_loss)
                cpts_loss2 = torch.sum(cpts_loss2)

            seg_loss      = seg_loss*loss_weight[0]
            cpts_loss     = cpts_loss*loss_weight[2]
            cpts_loss2    = cpts_loss2*loss_weight[3]
            loss          = seg_loss + cpts_loss + cpts_loss2

            ave_loss.update(loss.item())
            ave_seg_loss.update(seg_loss.item())
            # ave_bound_loss.update(seg_loss2.item())

            ave_Wing_loss.update(cpts_loss.item())
            ave_FL_loss.update(cpts_loss2.item())

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
    val_total_loss = ave_loss.average()
    val_segmentation_loss = ave_seg_loss.average()
    # val_boundary_loss = ave_bound_loss.average()
    val_Wing_loss = ave_Wing_loss.average()
    val_FL_loss = ave_FL_loss.average()

    mean_distance = total_distance/total_num_Present
    mpck = total_num_inThresh/total_num_Present.cpu().numpy()*100
    
    # total_num_Absent = total_num_points-total_num_Present
    # total_num_falseAbsent = total_num_Present -total_num_truePresent
    # total_num_falsePresent = total_num_Absent -total_num_trueAbsent
    # val_presence_accuracy = (total_num_truePresent+total_num_trueAbsent)/total_num_points*100
    # val_presence_precision = torch.tensor([total_num_truePresent/(total_num_truePresent+total_num_falsePresent), 
    #                                      total_num_trueAbsent/(total_num_trueAbsent+total_num_falseAbsent)])*100
    # val_presence_recall = torch.tensor([total_num_truePresent/total_num_Present,
    #                                   total_num_trueAbsent/total_num_Absent])*100

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('Loss/valid_loss', val_total_loss, global_steps)
    writer.add_scalar('Seg_loss/valid_segmentation_loss',
                      val_segmentation_loss, global_steps)
    # writer.add_scalar('Boundary_loss/valid_boundary_loss',
    #                   val_boundary_loss, global_steps)
    writer.add_scalar('Landmark_loss/valid_landmark_loss',
                      val_Wing_loss, global_steps)
    writer.add_scalar('Landmark_loss2/valid_landmark_loss2',
                      val_FL_loss, global_steps)
    writer.add_scalar('Mean_distance/valid_mDistance',
                      mean_distance, global_steps)
    writer.add_scalar('mIoU/valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('MPCK15/valid_MPCK15', mpck[-2], global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    
    
    msg = 'Valid_total_Loss: {:.6f}, Valid_seg_loss: {:.6f}, Valid_Wing_loss: {:.6f}, Valid_FL_loss: {:.6f}'.format(
            val_total_loss, val_segmentation_loss, val_Wing_loss, val_FL_loss)
    # msg = 'Valid_total_Loss: {:.6f}, Valid_seg_loss: {:.6f}, Valid_bound_loss:{:.6f}, Valid_Wing_loss: {:.6f}, Valid_FL_loss: {:.6f}'.format(
    #         val_total_loss, val_segmentation_loss, val_boundary_loss, val_Wing_loss, val_FL_loss)
    # msg = 'Valid_total_Loss: {:.6f}, Valid_seg_loss: {:.6f}, Valid_cpts_loss: {:.6f}, Valid_presence_loss: {:.6f}'.format(
    #         val_total_loss, val_segmentation_loss, val_centroid_loss, val_presence_loss)
    logging.info(msg)
    
    return val_total_loss, mean_IoU, IoU_array, accuracy, recall, precision, mean_distance, mpck
