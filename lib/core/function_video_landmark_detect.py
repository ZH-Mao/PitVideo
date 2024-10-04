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


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,trainloader, optimizer, model,
          Landmark_loss, Landmark_loss2, L1_smooth, writer_dict, device, loss_weight, scheduler=None):

    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_Wing_loss = AverageMeter()
    ave_FL_loss = AverageMeter()
    ave_L1_loss = AverageMeter()
    
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    # global_steps is the number of epoches rather than steps
    global_steps = writer_dict['train_global_steps']
    total_num_inThresh = np.zeros(4)  # 4 threshold [36, 72, 108, 144]
    total_num_points = 0
    total_distance = 0
    total_num_Present= 0 

    for i_iter, batch in enumerate(tqdm(trainloader)):
        images, _, cpts_gt, cpts_presence, _= batch
        images = rearrange(images, 'b l c h w -> (b l) c h w')
        cpts_gt = rearrange(cpts_gt, 'b l x y -> (b l) x y')
        cpts_presence = rearrange(cpts_presence, 'b l x y -> (b l) x y')
        images  = images.to(device)
        cpts_gt = cpts_gt.to(device)
        cpts_presence = cpts_presence.to(device)
        
        cpts_out = model(images)
        
        cpts_out = torch.reshape(cpts_out, (cpts_gt.size(0), cpts_gt.size(1), cpts_gt.size(2)))
        cpts_loss = Landmark_loss(cpts_out, cpts_gt)
        cpts_loss2 = Landmark_loss2(cpts_out, cpts_gt)
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
            
        
        cpts_out_t = rearrange(cpts_out, '(b l) x y -> b l x y', b=config.TRAIN.BATCH_SIZE_PER_GPU)
        cpts_out_t = torch.cat([cpts_out_t[:,1:], cpts_out_t[:,-1:]], dim=1)
        cpts_out_t = rearrange(cpts_out_t, 'b l x y -> (b l) x y')
        cpts_loss3 = L1_smooth(cpts_out, cpts_out_t)

       
        # calculate euclidean_distance between predicted and ground-truth landmarks
        cpts_presence = rearrange(cpts_presence, '(b l) x y -> b l x y', b=config.TRAIN.BATCH_SIZE_PER_GPU)[:,-1]
        cpts_out      = rearrange(cpts_out, '(b l) x y -> b l x y', b=config.TRAIN.BATCH_SIZE_PER_GPU)[:,-1]
        cpts_gt       = rearrange(cpts_gt, '(b l) x y -> b l x y', b=config.TRAIN.BATCH_SIZE_PER_GPU)[:,-1]
        squared_distance = torch.square(cpts_out - cpts_gt).detach() * cpts_presence
        squared_distance[:, :, 0] *= 1280**2
        squared_distance[:, :, 1] *= 720**2
        euclidean_distance = torch.sqrt(torch.sum(squared_distance, dim=2))
        # thresholds
        thresholds = [36, 72, 108, 144]
        for i, thresh in enumerate(thresholds):
            num_inThresh = ((euclidean_distance <= thresh) * cpts_presence[:, :, 0]).float()
            total_num_inThresh[i] += torch.sum(num_inThresh).item() 

        total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)
        total_num_Present += torch.sum(cpts_presence[:, :, 0])
        total_distance += torch.sum(euclidean_distance)

        cpts_loss     = cpts_loss*loss_weight[0]
        cpts_loss2    = cpts_loss2*loss_weight[1]
        cpts_loss3    = cpts_loss3*loss_weight[2]
        loss          = cpts_loss + cpts_loss2 + cpts_loss3
        
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
        ave_Wing_loss.update(cpts_loss.item())
        ave_FL_loss.update(cpts_loss2.item())
        ave_L1_loss.update(cpts_loss3.item())

        lr = base_lr * ((1-float(i_iter+cur_iters)/num_iters)**(0.9))
        
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr
            

    train_mdistance = total_distance/total_num_Present
    train_mpck = total_num_inThresh/total_num_Present.cpu().numpy()*100
    train_total_loss = ave_loss.average()
    train_Wing_loss = ave_Wing_loss.average()
    train_FL_loss = ave_FL_loss.average()
    train_L1_loss = ave_L1_loss.average()
    

    msg = 'Epoch: [{}/{}], Time: {:.2f}, ' \
        'lr: {:.6f}, Train_total_Loss: {:.6f}, Train_Wing_loss: {:.6f}, Train_FL_loss: {:.6f}, Train_L1_loss: {:.6f}'.format(
            epoch, num_epoch, batch_time.average(), optimizer.param_groups[1]['lr'], train_total_loss, train_Wing_loss, train_FL_loss, train_L1_loss
        )

    logging.info(msg)

    # Here we add_scalar every config.PRINT_FREQ.
    # Since in the same epoch, the global step is the same, when add_scalar, it will overwrite the previous one.
    writer.add_scalar('Loss/train_total_loss', train_total_loss, global_steps)
    writer.add_scalar('Landmark_loss/train_landmark_loss',
                      train_Wing_loss, global_steps)
    writer.add_scalar('Landmark_loss2/train_landmark_loss2',
                      train_FL_loss, global_steps)
    writer.add_scalar('Landmark_loss3/train_L1_loss', train_L1_loss, global_steps)
    writer.add_scalar('Mean_distance/train_mdistance',
                      train_mdistance, global_steps)
    writer.add_scalar('MPCK15/train_mpck15', train_mpck[-2], global_steps)
    writer.add_scalar('Learning_rate', optimizer.param_groups[1]['lr'], global_steps)

    writer_dict['train_global_steps'] = global_steps + 1
    return train_total_loss, train_mdistance, train_mpck


def validate(config, testloader, model, Landmark_loss, Landmark_loss2, L1_smooth, writer_dict, device, loss_weight):
    model.eval()
    ave_loss = AverageMeter()
    ave_Wing_loss = AverageMeter()
    ave_FL_loss = AverageMeter()
    ave_L1_loss = AverageMeter()
    total_num_inThresh = np.zeros(4)
    total_num_points = 0
    total_distance = 0
    total_num_Present= 0 

    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            images, _, cpts_gt, cpts_presence, _ = batch
            images = rearrange(images, 'b l c h w -> (b l) c h w')
            cpts_gt = rearrange(cpts_gt, 'b l x y -> (b l) x y')
            cpts_presence = rearrange(cpts_presence, 'b l x y -> (b l) x y')
            images = images.to(device)
            cpts_gt = cpts_gt.to(device)
            cpts_presence = cpts_presence.to(device)
            
            cpts_out = model(images)
            
            cpts_out = torch.reshape(cpts_out, (cpts_out.size(0), cpts_gt.size(1), cpts_gt.size(2)))
            cpts_loss = Landmark_loss(cpts_out, cpts_gt)
            cpts_loss2 = Landmark_loss2(cpts_out, cpts_gt)
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
            
            cpts_out_t = rearrange(cpts_out, '(b l) x y -> b l x y', b=config.TEST.BATCH_SIZE_PER_GPU)
            cpts_out_t = torch.cat([cpts_out_t[:,1:], cpts_out_t[:,-1:]], dim=1)
            cpts_out_t = rearrange(cpts_out_t, 'b l x y -> (b l) x y')
            cpts_loss3 = L1_smooth(cpts_out, cpts_out_t)
            
            # calculate euclidean_distance between predicted and ground-truth landmarks
            cpts_presence = rearrange(cpts_presence, '(b l) x y -> b l x y', b=config.TEST.BATCH_SIZE_PER_GPU)[:,-1]
            cpts_out = rearrange(cpts_out, '(b l) x y -> b l x y', b=config.TEST.BATCH_SIZE_PER_GPU)[:,-1]
            cpts_gt  = rearrange(cpts_gt, '(b l) x y -> b l x y', b=config.TEST.BATCH_SIZE_PER_GPU)[:,-1]
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
            

            cpts_loss     = cpts_loss*loss_weight[0]
            cpts_loss2    = cpts_loss2*loss_weight[1]
            cpts_loss3    = cpts_loss3*loss_weight[2]
            loss          = cpts_loss + cpts_loss2 + cpts_loss3

            ave_loss.update(loss.item())
            ave_Wing_loss.update(cpts_loss.item())
            ave_FL_loss.update(cpts_loss2.item())
            ave_L1_loss.update(cpts_loss3.item())

    mean_distance = total_distance/total_num_Present
    mpck = total_num_inThresh/total_num_Present.cpu().numpy()*100
    val_total_loss = ave_loss.average()
    val_Wing_loss = ave_Wing_loss.average()
    val_FL_loss = ave_FL_loss.average()
    val_L1_loss = ave_L1_loss.average()
    

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('Loss/valid_loss', val_total_loss, global_steps)
    writer.add_scalar('Landmark_loss/valid_landmark_loss',
                      val_Wing_loss, global_steps)
    writer.add_scalar('Landmark_loss2/valid_landmark_loss2',
                      val_FL_loss, global_steps)
    writer.add_scalar('Landmark_loss3/valid_landmark_loss3', val_L1_loss, global_steps)
    writer.add_scalar('Mean_distance/valid_mDistance',
                      mean_distance, global_steps)
    writer.add_scalar('MPCK15/valid_MPCK15', mpck[-2], global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    
    
    msg = 'Valid_total_Loss: {:.6f}, Valid_Wing_loss: {:.6f}, Valid_FL_loss: {:.6f}, Valid_L1_loss: {:.6f} '.format(
            val_total_loss, val_Wing_loss, val_FL_loss, val_L1_loss)
    logging.info(msg)
    
    return val_total_loss, mean_distance, mpck
