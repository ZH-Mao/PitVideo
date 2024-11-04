from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
# from PIL import Image
import albumentations as album
import torch
# import matplotlib as plt
import cv2
from PIL import Image
# from .bdl_dataloader import dist_map_transform


class PitDataset(Dataset):
    # Initialize the class
    def __init__(self, cfg=None, is_train=True, to_tensor=False) -> None:
        super().__init__()
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAIN_SET
            # self.csv_file ='image_centroid_fold1_train.csv'
        else:
            self.csv_file = cfg.DATASET.TEST_SET
            # self.csv_file = 'image_centroid_fold1_val.csv'
        # video clips 
        self.csv_file2 = cfg.DATASET.CLIPS
        self.is_train = is_train
        # self.transform = transform
        # self.preprocessing = preprocessing
        self.data_root = cfg.DATASET.ROOT
        self.csv_file_root = cfg.DATASET.CSV_FILE_ROOT
        self.image_root = cfg.DATASET.IMAGE_ROOT
        self.mask_root = cfg.DATASET.MASK_ROOT
        self.to_tensor = to_tensor

        # load annotations
        self.landmarks_frame = pd.read_excel(os.path.join(
            self.data_root, self.csv_file_root, self.csv_file))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # load vide clips' images
        self.video_frames = pd.read_excel(os.path.join(
            self.data_root, self.csv_file_root, self.csv_file2))

        self.transform_train = album.Compose([
            album.ShiftScaleRotate(
                shift_limit=(-0.2, 0.2),
                scale_limit=(-0.2, 0.3),
                rotate_limit=(-30, 30),
                always_apply=False,
                p=0.5),
            album.Resize(height=736, width=1280),
            album.ColorJitter(brightness=0.4, contrast=0.3,
                              saturation=0.3, hue=0.1, always_apply=False, p=0.5),
        ], keypoint_params=album.KeypointParams(format='xy', remove_invisible=False),
                                             additional_targets={
                                               'image4': 'image', 
                                            #    'image3': 'image', 
                                               'image2': 'image',
                                            #    'image1': 'image',
                                               'image': 'image',
                                               'keypoints4': 'keypoints',
                                            #    'keypoints3': 'keypoints',
                                               'keypoints2': 'keypoints',
                                            #    'keypoints1': 'keypoints',
                                               'keypoints': 'keypoints'})

        self.transform_val = album.Compose([
            album.Resize(height=736, width=1280),
        ], keypoint_params=album.KeypointParams(format='xy', remove_invisible=False), 
                                           additional_targets={
                                               'image4': 'image', 
                                            #    'image3': 'image', 
                                               'image2': 'image',
                                            #    'image1': 'image',
                                               'image': 'image',
                                               'keypoints4': 'keypoints',
                                            #    'keypoints3': 'keypoints',
                                               'keypoints2': 'keypoints',
                                            #    'keypoints1': 'keypoints',
                                               'keypoints': 'keypoints'})
        
        
        # # to use boundary loss - BDL library
        # self.disttransform = dist_map_transform([1, 1], 3)

    # Return the length of the dataset
    def __len__(self) -> int:
        return len(self.landmarks_frame)

    # Return the item at the given index
    def __getitem__(self, idx: int) -> tuple:      
        labeled_frame_name = self.landmarks_frame.iloc[idx, 1]        
        # find index of labeled_frame_name in video sequence
        matching_rows = self.video_frames.loc[self.video_frames['image'] == labeled_frame_name]
        frameID_in_video = matching_rows.index.tolist()
        frame_name = []
        
        frame_name.append(self.video_frames.iloc[frameID_in_video[0]-4, 1])
        image4 = cv2.imread(os.path.join(self.image_root, self.video_frames.iloc[frameID_in_video[0]-4, 1]))
        image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
        mask4 = cv2.imread(os.path.join(self.mask_root, self.video_frames.iloc[frameID_in_video[0]-4, 1]), cv2.IMREAD_GRAYSCALE)/120
        cpts4 = self.video_frames.iloc[frameID_in_video[0]-4, 2:10].values.astype('float').reshape(-1, 2)
        cpts_presence4 = np.float32(cpts4 != -100)
        cpts4 = cpts4*cpts_presence4
        cpts4[:, 0] = cpts4[:, 0]*1280
        cpts4[:, 1] = cpts4[:, 1]*720
        
        # frame_name.append(self.video_frames.iloc[frameID_in_video[0]-3, 1])
        # image3 = cv2.imread(os.path.join(self.image_root, self.video_frames.iloc[frameID_in_video[0]-3, 1]))
        # image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        # mask3 = cv2.imread(os.path.join(self.mask_root, self.video_frames.iloc[frameID_in_video[0]-3, 1]), cv2.IMREAD_GRAYSCALE)/120
        # cpts3 = self.video_frames.iloc[frameID_in_video[0]-3, 2:].values.astype('float').reshape(-1, 2)
        # cpts_presence3 = np.float32(cpts3 != -100)
        # cpts3 = cpts3*cpts_presence3
        # cpts3[:, 0] = cpts3[:, 0]*1280
        # cpts3[:, 1] = cpts3[:, 1]*720
        
        frame_name.append(self.video_frames.iloc[frameID_in_video[0]-2, 1])
        image2 = cv2.imread(os.path.join(self.image_root, self.video_frames.iloc[frameID_in_video[0]-2, 1]))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        mask2 = cv2.imread(os.path.join(self.mask_root, self.video_frames.iloc[frameID_in_video[0]-2, 1]), cv2.IMREAD_GRAYSCALE)/120
        cpts2 = self.video_frames.iloc[frameID_in_video[0]-2, 2:10].values.astype('float').reshape(-1, 2)
        cpts_presence2 = np.float32(cpts2 != -100)
        cpts2 = cpts2*cpts_presence2
        cpts2[:, 0] = cpts2[:, 0]*1280
        cpts2[:, 1] = cpts2[:, 1]*720
        
        # frame_name.append(self.video_frames.iloc[frameID_in_video[0]-1, 1])
        # image1 = cv2.imread(os.path.join(self.image_root, self.video_frames.iloc[frameID_in_video[0]-1, 1]))
        # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        # mask1 = cv2.imread(os.path.join(self.mask_root, self.video_frames.iloc[frameID_in_video[0]-1, 1]), cv2.IMREAD_GRAYSCALE)/120
        # cpts1 = self.video_frames.iloc[frameID_in_video[0]-1, 2:].values.astype('float').reshape(-1, 2)
        # cpts_presence1 = np.float32(cpts1 != -100)
        # cpts1 = cpts1*cpts_presence1
        # cpts1[:, 0] = cpts1[:, 0]*1280
        # cpts1[:, 1] = cpts1[:, 1]*720
        
        frame_name.append(self.video_frames.iloc[frameID_in_video[0], 1])
        image = cv2.imread(os.path.join(self.image_root, self.video_frames.iloc[frameID_in_video[0], 1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_root, self.video_frames.iloc[frameID_in_video[0], 1]), cv2.IMREAD_GRAYSCALE)/120
        cpts = self.video_frames.iloc[frameID_in_video[0], 2:10].values.astype('float').reshape(-1, 2)
        cpts_presence = np.float32(cpts != -100)
        cpts = cpts*cpts_presence
        cpts[:, 0] = cpts[:, 0]*1280
        cpts[:, 1] = cpts[:, 1]*720
        
        cpts_presence_sequence = torch.stack([torch.from_numpy(cpts_presence4), 
                                              torch.from_numpy(cpts_presence2), 
                                              torch.from_numpy(cpts_presence)], dim=0)

        # since we don't have masks and landmarks for video frames, we use the same masks and landmarks from labeled frame.
        # this is because we only stack images; masks and landmarks are not stacked.
        if self.is_train:
            sample = self.transform_train(
                image4=image4, image2=image2,image=image, 
                keypoints4=cpts4, keypoints2=cpts2,  keypoints=cpts)
            image4, image2, image = sample['image4'], sample['image2'], sample['image']
            cpts4, cpts2,  cpts = sample['keypoints4'],  sample['keypoints2'],  sample['keypoints']
            frames_sequence = torch.stack([torch.from_numpy(image4), torch.from_numpy(image2),  torch.from_numpy(image)], dim=0)
            mask_sequence = torch.stack([torch.from_numpy(mask4), torch.from_numpy(mask2), torch.from_numpy(mask)], dim=0)
            cpts_sequence = torch.stack([torch.from_numpy(np.array(cpts4)), torch.from_numpy(np.array(cpts2)), torch.from_numpy(np.array(cpts))], dim=0)
            if self.to_tensor:
                frames_sequence = (frames_sequence/255.0 - self.mean) / self.std
                frames_sequence = frames_sequence.permute(0, 3, 1, 2)
        else:
            sample = self.transform_val(
                image4=image4, image2=image2, image=image, 
                keypoints4=cpts4, keypoints2=cpts2, keypoints=cpts)
            image4, image2, image = sample['image4'], sample['image2'], sample['image']
            cpts4, cpts2, cpts = sample['keypoints4'], sample['keypoints2'], sample['keypoints']
            frames_sequence = torch.stack([torch.from_numpy(image4), torch.from_numpy(image2), torch.from_numpy(image)], dim=0)
            mask_sequence = torch.stack([torch.from_numpy(mask4), torch.from_numpy(mask2), torch.from_numpy(mask)], dim=0)
            cpts_sequence = torch.stack([torch.from_numpy(np.array(cpts4)), torch.from_numpy(np.array(cpts2)), torch.from_numpy(np.array(cpts))], dim=0)
            if self.to_tensor:
                frames_sequence = (frames_sequence/255.0 - self.mean) / self.std
                frames_sequence = frames_sequence.permute(0, 3, 1, 2)

        # Normalize coordinates
        cpts_sequence = cpts_sequence*cpts_presence_sequence/torch.from_numpy(np.array([1280, 736]))

        # # find points that excess the range of image after aug and replace them with [-100, -100]
        cpts_sequence[(cpts_sequence[:, :, 0] < 0) | (cpts_sequence[:, :, 0] > 1) | (cpts_sequence[:, :, 1] < 0) | (cpts_sequence[:, :, 1] > 1)] = 0.

        
        # # to use boundary loss - BDL library
        # dist_map_tensor4=self.disttransform(mask4)
        # dist_map_tensor3=self.disttransform(mask3)
        # dist_map_tensor2=self.disttransform(mask2)
        # dist_map_tensor1=self.disttransform(mask1)
        # dist_map_tensor=self.disttransform(mask)
        # dist_map_tensor_sequence = torch.stack([dist_map_tensor4, dist_map_tensor3, dist_map_tensor2, dist_map_tensor1, dist_map_tensor], dim=0)

        return frames_sequence, mask_sequence, cpts_sequence, cpts_presence_sequence, frame_name
        # return frames_sequence, mask_sequence, cpts_sequence, cpts_presence_sequence, frame_name, dist_map_tensor_sequence




if __name__ == '__main__':
    dataset = PitDataset(is_train=True)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    print(dataset[0][3].shape)