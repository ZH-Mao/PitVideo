# ------------------------------------------------------------------------------
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'  #GPU id
# os.environ["CUDA_LAUNCH_BLOCKING"]='1'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'

import argparse
import os
import pprint
# import shutil
# import sys
import glob

import logging
import time
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
import torch.optim
# from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
# import segmentation_models_pytorch as smp

from lib.config import config
from lib.config import update_config
# from core.criterion import CrossEntropy, OhemCrossEntropy
# from lib.core.bdl_losses2 import GeneralizedDice, SurfaceLoss, DiceLoss
# from utils.modelsummary import get_model_summary
# from utils.utils import create_logger, FullModel, get_rank
from lib.utils.utils import create_logger
# from lib.core.function_video import train, validate, test
# from lib.datasets.pitVideoDataset import PitDataset
# from lib.models.segland_hrnet_mstcn import HighResolutionNet
from lib.core.function_video_landmark_detect import train, validate
from lib.datasets.pitVideoDataset_3Masks import PitDataset
from lib.models.hrnet_mstcn_regression import HRNet_MSTCN_Regression
import random
from lib.core import mmwing_loss, focal_loss
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
from itertools import chain


seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)
    parser.add_argument('--cfg',
                        default=r'/workspace/projects/PitVideo/experiments/pituitary/video_hrnet_mstcn_landmark_w48_1stage_fold1.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument("--gpu", type=str, default='1')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # distributed = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:{}'.format(args.local_rank))

    # build model
    model = HRNet_MSTCN_Regression(config)
    model.init_weights(config.MODEL.PRETRAINED)              
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # # Test net output
    # dump_input = torch.rand((1, 5, 3, 736, 1280)).to(device)
    # seg, cpts = model(dump_input)

    train_dataset = PitDataset(config, is_train=True, to_tensor=True)
    test_dataset = PitDataset(config, is_train=False, to_tensor=True)
   
    # frames_sequence, mask_sequence, cpts_sequence, cpts_presence_sequence, labeled_frame_name = train_dataset[0]

 # Resample
    if config.TRAIN.RESAMPLE:
        target = torch.tensor([torch.sum(cpts_presence[:, 0])
                            for _, _, _, cpts_presence, _  in train_dataset])
        class_count = np.bincount(target)
        class_weights = 1./torch.tensor(class_count, dtype=float)
        weights = class_weights[target.long()]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True)

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=None)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=None)


    Landmark_loss = mmwing_loss.WingLoss()
    Landmark_loss2 = focal_loss.FocalLoss()
    L1_smooth = nn.SmoothL1Loss()
    loss_weight=torch.tensor(config.TRAIN.LOSS_WEIGHT).to(device)

    hrnet_params = list(chain(model.module.hrnet.parameters()))
    mstcn_regress_params = list(chain(model.module.mstcn_regression.parameters()))

    for param in hrnet_params:
        param.requires_grad = False
    for param in mstcn_regress_params:
        param.requires_grad = True

    optimizer = optim.SGD([
        {'params': hrnet_params, 'lr': 0.0},
        {'params': mstcn_regress_params, 'lr': config.TRAIN.LR}], 
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WD,
        nesterov=config.TRAIN.NESTEROV)

    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    epoch_iters = np.int32(train_dataset.__len__() /
                           config.TRAIN.BATCH_SIZE_PER_GPU)
    train_best_mpck15 = 0
    best_mpck15 = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            train_best_mpck15 = checkpoint['train_best_mpck15']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        train_total_loss, train_mdistance, train_mpck,\
                = train(config, epoch, config.TRAIN.END_EPOCH, epoch_iters, config.TRAIN.LR, num_iters,trainloader, optimizer, model, 
                        Landmark_loss, Landmark_loss2, L1_smooth, writer_dict, device, loss_weight)

        if train_mpck[-2] > train_best_mpck15:
            train_best_mpck15 = train_mpck[-2]
            train_old_models = glob.glob(os.path.join(final_output_dir, "train_best_model_*"))
            for train_old_model in train_old_models:
                os.remove(train_old_model)
            torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'train_best_model_epo{:03d}.pth'.format(epoch)))
                
        valid_loss, valid_mDistance, mpck = validate(config, testloader, model, Landmark_loss, Landmark_loss2, L1_smooth, writer_dict, device, loss_weight)
        
        if (mpck[-2] > best_mpck15):
                best_mpck15 = mpck[-2]
                val_old_models = glob.glob(os.path.join(final_output_dir, "val_best_model_*"))
                for val_old_model in val_old_models:
                    os.remove(val_old_model)
                torch.save(model.module.state_dict(),os.path.join(final_output_dir, 'val_best_model_epo{:03d}.pth'.format(epoch)))
            
        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'train_best_mpck15': train_best_mpck15,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))
            
        msg = 'Train==> Loss:{:.3f}, mDistance:{: 4.4f}, Best_MPCK15:{: 4.4f}'.format(train_total_loss, train_mdistance, train_best_mpck15)

        metric = 'Train_Metric==>  MPCK:{}'.format(train_mpck)
        logging.info(msg)
        logging.info(metric)


        msg = 'Val==> Loss:{:.3f}, mDistance:{: 4.4f}, Best_MPCK15:{: 4.4f}'.format(valid_loss, valid_mDistance, best_mpck15)

        metric = 'Val_Metric==> MPCK:{}'.format(mpck)
        logging.info(msg)
        logging.info(metric)

        if epoch == end_epoch - 1:
            torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'final_state.pth'))

            writer_dict['writer'].close()
            end = timeit.default_timer()
            logger.info('Hours: %d' % np.int32((end-start)/3600))
            logger.info('Done')

if __name__ == '__main__':
    main()
