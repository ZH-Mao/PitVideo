import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'  #GPU id
# global imports
import albumentations as album
import numpy as np
import os
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from ast import literal_eval
from efficientnet_pytorch import EfficientNet
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models

# strongly typed
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor
from typing import Dict
from typing import List
from typing import Tuple
from lib.config import config
from lib.datasets.pitVideoDataset_3Masks_painet import PitDataset
import argparse
from lib.config import update_config
from lib.core.function_overlays_make_video_painet import test
from lib.utils.utils import create_logger
import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Test PAINET')

    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)
    parser.add_argument('--cfg',
                        default=r'/home/zhehua/codes/PitVideo-Segment-Landmark/experiments/pituitary/PAINET_TEST_makevideo_fold1.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('--model',
                        default= r'/home/zhehua/codes/pit-anat-temp/outputs/models/mul/fold0_1.pth',
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
    aux_params = dict(
        pooling="avg",  # one of 'avg', 'max'
        dropout=0.3,  # dropout ratio, default is None
        activation="sigmoid",  # activation function, default is None
        classes= 6 * 2,  # define number of output labels
    )

    model = smp.UnetPlusPlus(
        encoder_name= "efficientnet-b3",
        classes=6,
        aux_params=aux_params,
    )

    model.load_state_dict(torch.load(config.MODEL.PRETRAINED, weights_only=True))
    model = model.to(device)
    
    # prepare data
    test_dataset = PitDataset(config, is_train=False, to_tensor=True)
    
    testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config.WORKERS,
    pin_memory=True)
    
    output_folder = 'PAINET_overlays_make_video'
    test(config, testloader, model, sv_dir=os.path.join(final_output_dir, output_folder), sv_pred=True, device=device, temp_length=3)

if __name__ == '__main__':
    main()