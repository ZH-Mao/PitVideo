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
from .bdl_dataloader import dist_map_transform

# Create a PitDataset class


class PitDataset(Dataset):
    # Initialize the class
    def __init__(self, cfg=None, is_train=True) -> None:
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

        # load annotations
        self.landmarks_frame = pd.read_csv(os.path.join(
            self.data_root, self.csv_file_root, self.csv_file))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # load vide clips' images
        self.video_frames = pd.read_csv(os.path.join(
            self.data_root, self.csv_file_root, self.csv_file2))

        self.transform_train = album.Compose([
            album.ShiftScaleRotate(
                shift_limit=(-0.2, 0.2),
                # scale_limit=(-0.1, 0.2),
                # rotate_limit=(-np.pi/6, np.pi/6),
                scale_limit=(-0.2, 0.3),
                rotate_limit=(-30, 30),
                always_apply=False,
                p=0.5),
            # album.Resize(height=736, width=1280),
            album.Resize(height=720, width=1296),
            album.ColorJitter(brightness=0.4, contrast=0.3,
                              saturation=0.3, hue=0.1, always_apply=False, p=0.5),
        ], keypoint_params=album.KeypointParams(format='xy', remove_invisible=False),
                                             additional_targets={
                                            #    'image4': 'image', 
                                            #    'image3': 'image', 
                                               'image2': 'image',
                                               'image1': 'image',
                                               'image': 'image'})

        self.transform_val = album.Compose([
            # album.Resize(height=736, width=1280),
            album.Resize(height=720, width=1296),
        ], keypoint_params=album.KeypointParams(format='xy', remove_invisible=False), 
                                           additional_targets={
                                            #    'image4': 'image', 
                                            #    'image3': 'image', 
                                               'image2': 'image',
                                               'image1': 'image',
                                               'image': 'image'})
        
        
        # # to use boundary loss - BDL library
        self.disttransform = dist_map_transform([1, 1], 3)

    # Return the length of the dataset
    def __len__(self) -> int:
        return len(self.landmarks_frame)

    # Return the item at the given index
    def __getitem__(self, idx: int) -> tuple:
        # image_path = os.path.join(self.image_root,
        #                           self.landmarks_frame.iloc[idx, 1])
        mask_path = os.path.join(self.mask_root,
                                 self.landmarks_frame.iloc[idx, 1].split('.')[0]+'.png')
        name = self.landmarks_frame.iloc[idx, 1].split('.')[0]

        # centroid points coordinates (cpts)
        cpts = self.landmarks_frame.iloc[idx, 2:].values
        cpts = cpts.astype('float').reshape(-1, 2)
        cpts_presence = np.float32(cpts != -100)
        cpts = cpts*cpts_presence
        cpts[:, 0] = cpts[:, 0]*1280
        cpts[:, 1] = cpts[:, 1]*720
      
        # mask_path = os.path.join(self.mask_root,
        #                          self.landmarks_frame.iloc[idx, 1].split('.')[0]+'.png')
        # name = self.landmarks_frame.iloc[idx, 1].split('.')[0]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        labeled_frame_name = self.landmarks_frame.iloc[idx, 1]        
        # find index of labeled_frame_name in video sequence
        matching_rows = self.video_frames.loc[self.video_frames['image'] == labeled_frame_name]
        frameID_in_video = matching_rows.index.tolist()
        

        # image4 = cv2.imread(os.path.join(self.image_root,
        #                         self.video_frames.iloc[frameID_in_video[0]-4, 0]))
        # image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
        
        # image3 = cv2.imread(os.path.join(self.image_root,
        #                         self.video_frames.iloc[frameID_in_video[0]-3, 0]))
        # image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        
        image2 = cv2.imread(os.path.join(self.image_root,
                                self.video_frames.iloc[frameID_in_video[0]-2, 0]))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        image1 = cv2.imread(os.path.join(self.image_root,
                                self.video_frames.iloc[frameID_in_video[0]-1, 0]))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        image = cv2.imread(os.path.join(self.image_root,
                                self.video_frames.iloc[frameID_in_video[0], 0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # since we don't have masks and landmarks for video frames, we use the same masks and landmarks from labeled frame.
        # this is because we only stack images; masks and landmarks are not stacked.
        if self.is_train:
            # sample = self.transform_train(
            #     image4=image4, image3=image3, image2=image2, image1=image1,image=image,  mask=mask, keypoints=cpts)
            # image4, image3, image2, image1, image, mask, cpts = sample['image4'],sample['image3'],sample['image2'],sample['image1'],sample['image'], sample['mask'], sample["keypoints"]
            # frames_sequence = torch.stack([torch.from_numpy(image4), torch.from_numpy(image3), torch.from_numpy(image2), torch.from_numpy(image1), torch.from_numpy(image)], dim=0)
            sample = self.transform_train(
                image2=image2, image1=image1,image=image,  mask=mask, keypoints=cpts)
            image2, image1, image, mask, cpts = sample['image2'],sample['image1'],sample['image'], sample['mask'], sample["keypoints"]
            frames_sequence = torch.stack([torch.from_numpy(image2), torch.from_numpy(image1), torch.from_numpy(image)], dim=0)
            
            # frames_sequence = frames_sequence.astype(np.float32)
            frames_sequence = (frames_sequence/255.0 - self.mean) / self.std
            frames_sequence = frames_sequence.permute(0, 3, 1, 2)
        else:
            # sample = self.transform_val(image4=image4, image3=image3, image2=image2, image1=image1,image=image,  mask=mask, keypoints=cpts)
            # image4, image3, image2, image1, image, mask, cpts = sample['image4'],sample['image3'],sample['image2'],sample['image1'],sample['image'], sample['mask'], sample["keypoints"]
            # frames_sequence = torch.stack([torch.from_numpy(image4), torch.from_numpy(image3), torch.from_numpy(image2), torch.from_numpy(image1), torch.from_numpy(image)], dim=0)
            sample = self.transform_val(
                image2=image2, image1=image1,image=image,  mask=mask, keypoints=cpts)
            image2, image1, image, mask, cpts = sample['image2'],sample['image1'],sample['image'], sample['mask'], sample["keypoints"]
            frames_sequence = torch.stack([torch.from_numpy(image2), torch.from_numpy(image1), torch.from_numpy(image)], dim=0)
            # frames_sequence = frames_sequence.astype(np.float32)
            frames_sequence = (frames_sequence/255.0 - self.mean) / self.std
            frames_sequence = frames_sequence.permute(0, 3, 1, 2)

        # Normalize image
        cpts = cpts*cpts_presence
        # cpts[:, 0] = cpts[:, 0]/1280
        # cpts[:, 1] = cpts[:, 1]/736
        cpts[:, 0] = cpts[:, 0]/1296
        cpts[:, 1] = cpts[:, 1]/720

        # find points that excess the range of image after aug and replace them with [-100, -100]
        condition_first_column = (cpts[:, 0] > 1) | (cpts[:, 0] < 0)
        condition_second_column = (cpts[:, 1] > 1) | (cpts[:, 1] < 0)
        rows_to_change = np.where(
            condition_first_column | condition_second_column)
        cpts[rows_to_change] = 0.

        # Replace coordinates that are original absent with [-100,-100]
        cpts_absence = np.float32(cpts_presence == 0)
        cpts = cpts-0.*cpts_absence
        cpts_presence = np.float32(cpts != 0.)
        # image = image.astype(np.float32)
        # image = (image/255.0 - self.mean) / self.std
        # image = image.transpose([2, 0, 1])
        # image = torch.Tensor(image)
        mask = torch.Tensor(mask)
        cpts = torch.Tensor(cpts)
        cpts_presence = torch.Tensor(cpts_presence)
        
        # to use boundary loss - BDL library
        dist_map_tensor=self.disttransform(mask)

        return frames_sequence, mask, cpts, cpts_presence, name, dist_map_tensor
        # return image, mask, cpts, cpts_presence, name



if __name__ == '__main__':
    dataset = PitDataset(is_train=True)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    print(dataset[0][3].shape)