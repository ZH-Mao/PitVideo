# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU id
import random
from lib.models.seg_hrnet_v4_1 import HighResolutionNet
from lib.datasets.pitVideoDataset_3Masks import PitDataset
from lib.utils.utils import create_logger
from lib.core.function_consistency_consecutiveIOUbased_test import test
from lib.config import update_config
from lib.config import config
# import segmentation_models_pytorch as smp
# from tensorboardX import SummaryWriter
import torch.optim
import torch
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pprint
import argparse


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
    # parser.add_argument('--model',
    #                     default= r'/home/zhehua/data/Results/pituitary/video_hrnet_convlstm_w48_2stage_5loss_fold1/val_best_model_epo130.pth',
    #                     help='trained model file',
    #                     type=str)
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
    model.init_weights(config.MODEL.PRETRAINED)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.to(device), dump_input.to(device)))

    # if config.TEST.MODEL_FILE:
    #     model_state_file = config.TEST.MODEL_FILE
    # else:
    #     model_state_file = os.path.join(final_output_dir,
    #                                     'best_mIoU.pth')
    # logger.info('=> loading model from {}'.format(model_state_file))

    # prepare data
    test_dataset = PitDataset(config, is_train=False, to_tensor=True)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    start = timeit.default_timer()
    
    output_folder = 'PitImage_convlstm_consecutiveIOU_results'
    test(config, testloader, model, sv_dir=os.path.join(final_output_dir, output_folder), sv_pred=True, device=device)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int32((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
