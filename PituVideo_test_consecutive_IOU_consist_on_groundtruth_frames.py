# ------------------------------------------------------------------------------
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'  #GPU id
# os.environ["CUDA_LAUNCH_BLOCKING"]='1'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'

import argparse
import os
import pprint
# import shutil
# import sys

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

# import tools._init_paths
# import models
# import datasets
from lib.config import config
from lib.config import update_config
# from core.criterion import CrossEntropy, OhemCrossEntropy
# from lib.core.bdl_losses import GeneralizedDice, SurfaceLoss, DiceLoss
# from utils.modelsummary import get_model_summary
# from utils.utils import create_logger, FullModel, get_rank
from lib.utils.utils import create_logger
# from lib.core.function_video import train, validate, test
# from lib.datasets.pitVideoDataset import PitDataset
# from lib.models.segland_hrnet_mstcn import HighResolutionNet
from lib.core.function_consistency_consecutiveIOUbased_test_perframe import test
from lib.datasets.pitVideoDataset_3Masks import PitDataset
from lib.models.segland_hrnet_convLSTM import HighResolutionNet
import random
# from lib.core import mmwing_loss, focal_loss
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# from itertools import chain


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
                        default=r'/home/zhehua/codes/PitVideo-Segment-Landmark/experiments/pituitary/video_hrnet_convlstm_w48_2stage_5loss_fold1.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('--model',
                        default= r'/home/zhehua/data/Results/pituitary/video_hrnet_convlstm_w48_2stage_5loss_fold1/val_best_model3_epo196.pth',
                        help='trained model file',
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

    logger, final_output_dir = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    model = HighResolutionNet(config)
    # model.init_weights(config.MODEL.PRETRAINED)
    model.init_weights(args.model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # prepare data
    test_dataset = PitDataset(config, is_train=False, to_tensor=True)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2, #batch size has to be 1, otherwise the model will not work
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    start = timeit.default_timer()

    # output_file_excel = 'tempconsisnet.xlsx'
    test(config, testloader, model, sv_dir=final_output_dir, sv_pred=False, device=device, temp_length=3)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int32((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
