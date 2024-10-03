# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from ..utils.utils import AverageMeter
from ..utils.utils import get_confusion_matrix
from ..utils.utils import adjust_learning_rate
from ..utils.utils import get_world_size, get_rank
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw


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


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
          trainloader, optimizer, model, Seg_loss, Seg_loss2, Landmark_loss, Landmark_presence_loss, writer_dict, device, scheduler, seg_act=False, land_act=False):

    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_seg_loss = AverageMeter()
    ave_cpts_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    # global_steps is the number of epoches rather than steps
    global_steps = writer_dict['train_global_steps']
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    total_num_inThresh = 0
    total_num_points = 0
    total_distance = 0
    total_num_Present= 0 
    total_num_Absent=0
    total_num_truePresent = 0
    total_num_trueAbsent = 0

    for i_iter, batch in enumerate(tqdm(trainloader)):
        images, labels, cpts_gt, cpts_presence, _ = batch
        size = labels.size()
        images = images.to(device)
        labels = labels.long().to(device)
        cpts_gt = cpts_gt.to(device)
        cpts_presence = cpts_presence.to(device)
        total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)

        seg_out, cpts_out, cpts_presence_out = model(images)
        if seg_act:
            seg_out = F.softmax(seg_out, dim=1)
        if land_act:
            cpts_presence_out = F.sigmoid(cpts_presence_out)
        ph, pw = seg_out.size(2), seg_out.size(3)
        h, w = labels.size(1), labels.size(2)
        if ph != h or pw != w:
            seg_out = F.interpolate(input=seg_out, size=(h, w), mode='bilinear')

        seg_loss = Seg_loss(seg_out, labels)
        # seg_loss2 = Seg_loss2(torch.sigmoid(seg_out), labels.unsqueeze(1))
        seg_loss2 = Seg_loss2(seg_out, labels)
        cpts_out = torch.reshape(
            cpts_out, (cpts_out.size(0), cpts_gt.size(1), cpts_gt.size(2)))
        cpts_loss = Landmark_loss(cpts_out, cpts_gt)
        presence_loss = Landmark_presence_loss(cpts_presence_out, cpts_presence[:,:,0])

        # calculate euclidean_distance between predicted and ground-truth landmarks
        cpts_present_loss = cpts_loss * cpts_presence
        squared_distance = torch.zeros_like(cpts_present_loss)
        squared_distance[:, :, 0] = cpts_present_loss[:, :, 0]*(1280**2)
        squared_distance[:, :, 1] = cpts_present_loss[:, :, 1]*(736**2)
        euclidean_distance = torch.sum(
            squared_distance, dim=(2), keepdim=True)
        euclidean_distance = torch.sqrt(euclidean_distance.squeeze(dim=2))
        # calculate how many points are within 144 pixels from their corresponding ground truth
        num_inThresh = ((euclidean_distance >= 0) & (
            euclidean_distance <= 144)).float()
        num_inThresh = num_inThresh*cpts_presence[:, :, 0]
        total_num_inThresh += torch.sum(num_inThresh)
        total_num_Present += torch.sum(cpts_presence[:, :, 0])
        total_distance += torch.sum(euclidean_distance)
        
        # calculate the number of presence
        pre_presence = torch.where(torch.sigmoid(cpts_presence_out).cpu() < torch.tensor(0.48), torch.tensor(0), torch.tensor(1)).to(device)
        total_num_truePresent += torch.sum((pre_presence+cpts_presence[:, :, 0].to(torch.uint8)==2).long())
        total_num_trueAbsent += torch.sum((pre_presence+cpts_presence[:, :, 0].to(torch.uint8)==0).long())

        # cpts_loss = cpts_loss.mean()
        cpts_loss = cpts_loss * cpts_presence

        if torch.sum(cpts_presence) > 0:
            cpts_loss = torch.sum(cpts_loss) / torch.sum(cpts_presence)
        else:
            cpts_loss = torch.sum(cpts_loss)

        if torch.isnan(cpts_loss):
            print("cpts_loss is nan")

        if torch.isnan(seg_loss):
            print("seg_loss is nan")

        loss = seg_loss+0.1*seg_loss2+cpts_loss+presence_loss
        if torch.isnan(loss):
            print("loss is nan")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_seg_loss.update(seg_loss.item())
        ave_cpts_loss.update(cpts_loss.item())

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
    train_accuracy = tp.sum()/pos.sum()
    train_recall = (tp/np.maximum(1.0, pos))
    train_precision = (tp/np.maximum(1.0, res))
    train_IoU = (tp / np.maximum(1.0, pos + res - tp))
    train_mIoU = train_IoU[-2:].mean()
    train_mdistance = total_distance/total_num_Present
    train_mpck20 = total_num_inThresh/total_num_Present
    train_total_loss = ave_loss.average()
    train_segmentation_loss = ave_seg_loss.average()
    train_centroid_loss = ave_cpts_loss.average()
    
    total_num_Absent = total_num_points-total_num_Present
    total_num_falseAbsent = total_num_Present -total_num_truePresent
    total_num_falsePresent = total_num_Absent -total_num_trueAbsent
    train_presence_accuracy = (total_num_truePresent+total_num_trueAbsent)/total_num_points
    train_presence_precision = torch.tensor([total_num_truePresent/(total_num_truePresent+total_num_falsePresent), 
                                         total_num_trueAbsent/(total_num_trueAbsent+total_num_falseAbsent)])
    train_presence_recall = torch.tensor([total_num_truePresent/total_num_Present,
                                      total_num_trueAbsent/total_num_Absent])
    
    

    # Decay learning rate
    scheduler.step()
    for pg in optimizer.param_groups:
        if pg['lr'] < 1e-4:
            pg['lr'] = 1e-4

    msg = 'Epoch: [{}/{}], Time: {:.2f}, ' \
        'lr: {:.6f}, Train_total_Loss: {:.6f}, Train_seg_loss: {:.6f}, Train_cpts_loss: {:.6f}'.format(
            epoch, num_epoch, batch_time.average(
            ), optimizer.param_groups[0]['lr'], train_total_loss,
            train_segmentation_loss, train_centroid_loss
        )
    logging.info(msg)

    # Here we add_scalar every config.PRINT_FREQ.
    # Since in the same epoch, the global step is the same, when add_scalar, it will overwrite the previous one.
    writer.add_scalar('Loss/train_total_loss', train_total_loss, global_steps)
    writer.add_scalar('Seg_loss/train_segmentation_loss',
                      train_segmentation_loss, global_steps)
    writer.add_scalar('Landmark_loss/train_landmark_loss',
                      train_centroid_loss, global_steps)
    writer.add_scalar('Mean_distance/train_mdistance',
                      train_mdistance, global_steps)
    writer.add_scalar('mIoU/train_mIoU', train_mIoU, global_steps)
    writer.add_scalar('MPCK20/train_mpck20', train_mpck20, global_steps)

    writer_dict['train_global_steps'] = global_steps + 1
    return train_total_loss, train_mIoU, train_IoU, train_accuracy, train_recall, train_precision, train_mdistance, train_mpck20,\
           train_presence_accuracy, train_presence_precision, train_presence_recall


def validate(config, testloader, model, Seg_loss, Seg_loss2, Landmark_loss, Landmark_presence_loss, writer_dict, device):
    model.eval()
    ave_loss = AverageMeter()
    ave_seg_loss = AverageMeter()
    ave_cpts_loss = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    total_num_inThresh = 0
    total_num_points = 0
    total_distance = 0
    total_num_Present= 0 
    total_num_Absent=0
    total_num_truePresent = 0
    total_num_trueAbsent = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, label, cpts_gt, cpts_presence, _ = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)
            cpts_gt = cpts_gt.to(device)
            cpts_presence = cpts_presence.to(device)
            total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)

            seg_pre, cpts_pre, cpts_presence_pre = model(image)
            pred = F.interpolate(input=seg_pre, size=(
                size[-2], size[-1]), mode='bilinear')
            seg_loss = Seg_loss(pred, label)
            # seg_loss2=Seg_loss2(torch.sigmoid(pred), label.unsqueeze(1))
            seg_loss2=Seg_loss2(pred, label)

            cpts_pre = torch.reshape(
                cpts_pre, (cpts_pre.size(0), cpts_gt.size(1), cpts_gt.size(2)))
            cpts_loss = Landmark_loss(cpts_pre, cpts_gt)
            presence_loss = Landmark_presence_loss(cpts_presence_pre, cpts_presence[:,:,0])
            

            # calculate euclidean_distance between predicted and ground-truth landmarks
            cpts_present_loss = cpts_loss * cpts_presence
            squared_distance = torch.zeros_like(cpts_present_loss)
            squared_distance[:, :, 0] = cpts_present_loss[:, :, 0]*(1280**2)
            squared_distance[:, :, 1] = cpts_present_loss[:, :, 1]*(736**2)
            euclidean_distance = torch.sum(
                squared_distance, dim=(2), keepdim=True)
            euclidean_distance = torch.sqrt(euclidean_distance.squeeze(dim=2))
            # calculate how many points are within 144 pixels from their corresponding ground truth
            num_inThresh = ((euclidean_distance >= 0) & (
                euclidean_distance <= 144)).float()
            num_inThresh = num_inThresh*cpts_presence[:, :, 0]
            total_num_inThresh += torch.sum(num_inThresh)
            total_num_Present += torch.sum(cpts_presence[:, :, 0])
            total_distance += torch.sum(euclidean_distance)
            # if torch.sum(cpts_presence[:,:,0]) > 0:
            #     euclidean_distance = torch.sum(euclidean_distance) / torch.sum(cpts_presence[:,:,0])
            #     ave_distance.update(euclidean_distance.item())
            # else:
            #     euclidean_distance = torch.sum(euclidean_distance)
            #     ave_distance.update(euclidean_distance.item(), weight=0)
            
            # calculate the number of presence
            pre_presence = torch.where(torch.sigmoid(cpts_presence_pre).cpu() < torch.tensor(0.48), torch.tensor(0), torch.tensor(1)).to(device)
            total_num_truePresent += torch.sum((pre_presence+cpts_presence[:, :, 0].to(torch.uint8)==2).long())
            total_num_trueAbsent += torch.sum((pre_presence+cpts_presence[:, :, 0].to(torch.uint8)==0).long())

            cpts_loss = cpts_loss * cpts_presence
            if torch.sum(cpts_presence) > 0:
                cpts_loss = torch.sum(cpts_loss) / torch.sum(cpts_presence)
            else:
                cpts_loss = torch.sum(cpts_loss)

            cpts_loss = cpts_loss.mean()

            loss = seg_loss+0.1*seg_loss2+cpts_loss+presence_loss

            ave_loss.update(loss.item())
            ave_seg_loss.update(seg_loss.item())
            ave_cpts_loss.update(cpts_loss.item())

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    accuracy = tp.sum()/pos.sum()
    recall = (tp/np.maximum(1.0, pos))
    precision = (tp/np.maximum(1.0, res))
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array[-2:].mean()
    val_total_loss = ave_loss.average()
    val_segmentation_loss = ave_seg_loss.average()
    val_centroid_loss = ave_cpts_loss.average()

    mean_distance = total_distance/total_num_Present
    mpck20 = total_num_inThresh/total_num_Present
    
    total_num_Absent = total_num_points-total_num_Present
    total_num_falseAbsent = total_num_Present -total_num_truePresent
    total_num_falsePresent = total_num_Absent -total_num_trueAbsent
    val_presence_accuracy = (total_num_truePresent+total_num_trueAbsent)/total_num_points
    val_presence_precision = torch.tensor([total_num_truePresent/(total_num_truePresent+total_num_falsePresent), 
                                         total_num_trueAbsent/(total_num_trueAbsent+total_num_falseAbsent)])
    val_presence_recall = torch.tensor([total_num_truePresent/total_num_Present,
                                      total_num_trueAbsent/total_num_Absent])

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('Loss/valid_loss', val_total_loss, global_steps)
    writer.add_scalar('Seg_loss/valid_segmentation_loss',
                      val_segmentation_loss, global_steps)
    writer.add_scalar('Landmark_loss/valid_landmark_loss',
                      val_centroid_loss, global_steps)
    writer.add_scalar('Mean_distance/valid_mDistance',
                      mean_distance, global_steps)
    writer.add_scalar('mIoU/valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('MPCK20/valid_MPCK20', mpck20, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return val_total_loss, mean_IoU, IoU_array, accuracy, recall, precision, mean_distance, mpck20,\
        val_presence_accuracy, val_presence_precision, val_presence_recall


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _ = batch
            name = 'testfold1'
            size = label.size()
            # pred = test_dataset.multi_scale_inference(
            #             model,
            #             image,
            #             scales=config.TEST.SCALE_LIST,
            #             flip=config.TEST.FLIP_TEST)
            pred, cpts = model(image)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]),
                                  mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()  # accuracy
    mean_acc = (tp/np.maximum(1.0, pos)).mean()  # recall
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, testloader, model,
         sv_dir='', sv_pred=True, device=None):
    model.eval()
    total_num_pre_presence = 0
    total_num_pre_presence_true = 0
    total_num_points = 0
        
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, label, cpts_gt, cpts_presence, name = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)
            cpts_gt = cpts_gt.to(device)
            cpts_presence = cpts_presence.to(device)

            seg_pre, cpts_pre, cpts_presence_pre = model(image)
            pred = F.interpolate(input=seg_pre, size=(
                size[-2], size[-1]), mode='bilinear')

            cpts_pre = torch.reshape(
                cpts_pre, (cpts_pre.size(0), cpts_gt.size(1), cpts_gt.size(2)))

            # cpts_pre= cpts_pre*cpts_presence
            total_num_points += torch.sum(cpts_presence[:, :, 0])
            pre_presence = torch.where(torch.sigmoid(cpts_presence_pre).cpu() < torch.tensor(0.48), torch.tensor(0), torch.tensor(1)).to(device)
            total_num_pre_presence_true += torch.sum((pre_presence+cpts_presence[:, :, 0].to(torch.uint8)==2).long())
            total_num_pre_presence += torch.sum(pre_presence)
            pre_presence = pre_presence.unsqueeze(2).expand(-1, -1, 2)
            cpts_pre = cpts_pre*pre_presence
            

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            pred = pred.cpu().numpy().copy()
            ori_image = image.cpu().numpy().copy()
            ground_truth_landmarks = cpts_gt.cpu().numpy().copy()
            predicted_landmarks = cpts_pre.cpu().numpy().copy()
            label = label.cpu().numpy().copy()
            pred = np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                for i in range(pred.shape[0]):
                    ori_image = (
                        (ori_image[0].transpose(1, 2, 0) * std) + mean)*255
                    ori_image = Image.fromarray(np.uint8(ori_image))
                    result_image = ori_image.copy()
                    draw = ImageDraw.Draw(result_image)
                    # Define colors
                    transparent = (0, 0, 0, 0)  # Transparent
                    green = (255, 255, 0, 200)  # Green with half transparency
                    red = (0, 0, 255, 200)  # Red with half transparency
                    # Overlay ground truth mask
                    ground_truth_mask = np.array(label[0], dtype=np.uint8)
                    mask_image = Image.new(
                        "RGBA", result_image.size, transparent)
                    for class_id in [1, 2]:
                        mask = (ground_truth_mask == class_id).astype(
                            np.uint8) * 200
                        mask_image.paste(
                            green, (0, 0), Image.fromarray(mask).convert("L"))
                    result_image.paste(mask_image, (0, 0), mask_image)
                    # Overlay predicted mask
                    predicted_mask = np.array(pred[0], dtype=np.uint8)
                    mask_image = Image.new(
                        "RGBA", result_image.size, transparent)
                    for class_id in [1, 2]:
                        mask = (predicted_mask == class_id).astype(
                            np.uint8) * 200
                        mask_image.paste(
                            red, (0, 0), Image.fromarray(mask).convert("L"))
                    result_image.paste(mask_image, (0, 0), mask_image)
                    # Overlay ground truth landmarks as green circles
                    for landmark in ground_truth_landmarks[0]:
                        x, y = landmark
                        if x != 0 and y != 0:
                            draw.ellipse(
                                [(x*1280 - 10, y*736 - 10), (x*1280 + 10, y*736 + 10)], fill=(0, 255, 0, 200))
                    # Overlay predicted landmarks as red crosses
                    for landmark in predicted_landmarks[0]:
                        x, y = landmark
                        if x != 0 and y != 0:
                            draw.line([(x*1280 - 10, y*736 - 10), (x*1280 + 10,
                                      y*736 + 10)], fill=(0, 255, 255, 200), width=6)
                            draw.line([(x*1280 + 10, y*736 - 10), (x*1280 - 10,
                                      y*736 + 10)], fill=(0, 255, 255, 200), width=6)
                result_image.save(os.path.join(sv_path, name[i]+'.png'))
        logging.info("{}/{} of predicted present landmarks are accurate, in total {} landmarks are really present".format(total_num_pre_presence_true,total_num_pre_presence, total_num_points))
