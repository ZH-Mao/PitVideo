from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import albumentations as album
import torch
import cv2
from PIL import Image

class PitDataset(Dataset):
    def __init__(self, cfg=None, is_train=True, to_tensor=False) -> None:
        super().__init__()
        # Select the annotation csv file based on the training mode
        if is_train:
            self.csv_file = cfg.DATASET.TRAIN_SET
        else:
            self.csv_file = cfg.DATASET.TEST_SET
        self.csv_file2 = cfg.DATASET.CLIPS
        self.is_train = is_train
        self.data_root = cfg.DATASET.ROOT
        self.csv_file_root = cfg.DATASET.CSV_FILE_ROOT
        self.image_root = cfg.DATASET.IMAGE_ROOT
        self.mask_root = cfg.DATASET.MASK_ROOT
        self.to_tensor = to_tensor

        # Load annotations and video clip information
        self.landmarks_frame = pd.read_excel(os.path.join(
            self.data_root, self.csv_file_root, self.csv_file))
        self.video_frames = pd.read_excel(os.path.join(
            self.data_root, self.csv_file_root, self.csv_file2))

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Set the relative positions for processing frames (default is 3 frames: [-4, -2, 0])
        self.frame_offsets = cfg.DATASET.FRAME_OFFSETS if hasattr(cfg.DATASET, 'FRAME_OFFSETS') else [-4, -2, 0]

        # Construct additional_targets for albumentations:
        # The main frame uses the default keys, while other frames are named based on the absolute value (e.g., -4 -> "image4", "mask4", "keypoints4")
        additional_targets = {}
        for offset in self.frame_offsets:
            if offset != 0:
                key_im = f"image{abs(offset)}"
                key_mask = f"mask{abs(offset)}"
                key_kpt = f"keypoints{abs(offset)}"
                additional_targets[key_im] = "image"
                additional_targets[key_mask] = "mask"
                additional_targets[key_kpt] = "keypoints"

        # Define augmentation transforms for training and validation
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
           additional_targets=additional_targets)

        self.transform_val = album.Compose([
            album.Resize(height=736, width=1280),
        ], keypoint_params=album.KeypointParams(format='xy', remove_invisible=False),
           additional_targets=additional_targets)

    def __len__(self) -> int:
        return len(self.landmarks_frame)

    def __getitem__(self, idx: int) -> tuple:
        # Obtain the labeled frame name; typically the annotation frame name is in the second column of landmarks_frame
        labeled_frame_name = self.landmarks_frame.iloc[idx, 1]
        # Find the corresponding frame in the video clip
        matching_rows = self.video_frames.loc[self.video_frames['image'] == labeled_frame_name]
        frameID_in_video = matching_rows.index.tolist()
        if len(frameID_in_video) == 0:
            raise IndexError(f"Could not find a matching frame in the video frames for: {labeled_frame_name}")

        # Build a dictionary for albumentations containing images, masks, and keypoints for all frames
        transform_args = {}
        # Save the original keypoint presence information (augmentation does not process this automatically)
        kpt_presence_dict = {}
        # Save the filenames of all frames
        frame_name_list = []

        for offset in self.frame_offsets:
            ref_index = frameID_in_video[0] + offset
            # Boundary check
            if ref_index < 0 or ref_index >= len(self.video_frames):
                raise IndexError(f"Calculated frame index {ref_index} is out of range for video frames.")
            row = self.video_frames.iloc[ref_index]
            # Generate different keys based on the offset: the labeled frame (offset==0) uses the default keys
            if offset == 0:
                img_key = "image"
                mask_key = "mask"
                kpt_key = "keypoints"
            else:
                img_key = f"image{abs(offset)}"
                mask_key = f"mask{abs(offset)}"
                kpt_key = f"keypoints{abs(offset)}"

            # Read the image and convert it to RGB
            img_path = os.path.join(self.image_root, row[1])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transform_args[img_key] = img

            # Read the mask and normalize it by dividing by 120
            mask_path = os.path.join(self.mask_root, row[1])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            mask = mask / 120.0
            transform_args[mask_key] = mask

            # Read keypoints; assume the keypoint data occupies columns 3 to 10 (i.e., indices 2 to 9)
            kp = row[2:10].values.astype('float').reshape(-1, 2)
            presence = (kp != -100).astype(np.float32)
            kp = kp * presence
            # Convert the normalized keypoint values to absolute pixel coordinates (assuming image width 1280 and height 720 or 736, as per the original code)
            kp[:, 0] = kp[:, 0] * 1280
            kp[:, 1] = kp[:, 1] * 720
            transform_args[kpt_key] = kp
            kpt_presence_dict[kpt_key] = presence

            frame_name_list.append(row[1])

        # Apply the augmentation transforms simultaneously to all frames
        if self.is_train:
            sample = self.transform_train(**transform_args)
            # Resize masks to ensure they match the dimensions (1280, 720); note that cv2.resize expects size in (width, height)
            for offset in self.frame_offsets:
                mask_key = "mask" if offset == 0 else f"mask{abs(offset)}"
                sample[mask_key] = cv2.resize(sample[mask_key], (1280, 720), interpolation=cv2.INTER_NEAREST)
        else:
            sample = self.transform_val(**transform_args)

        # Construct sequences of images, masks, and keypoints following the order in frame_offsets
        frames_list = []
        mask_list = []
        cpts_list = []
        cpts_presence_list = []
        for offset in self.frame_offsets:
            if offset == 0:
                img_key = "image"
                mask_key = "mask"
                kpt_key = "keypoints"
            else:
                img_key = f"image{abs(offset)}"
                mask_key = f"mask{abs(offset)}"
                kpt_key = f"keypoints{abs(offset)}"
            frames_list.append(torch.from_numpy(sample[img_key]))
            mask_list.append(torch.from_numpy(sample[mask_key]))
            cpts_list.append(torch.from_numpy(np.array(sample[kpt_key])))
            cpts_presence_list.append(torch.from_numpy(kpt_presence_dict[kpt_key]))

        frames_sequence = torch.stack(frames_list, dim=0)
        mask_sequence = torch.stack(mask_list, dim=0)
        cpts_sequence = torch.stack(cpts_list, dim=0)
        cpts_presence_sequence = torch.stack(cpts_presence_list, dim=0)

        # If specified, convert to tensor, normalize the images, and permute the dimensions to [frames, channels, height, width]
        if self.to_tensor:
            frames_sequence = (frames_sequence.float() / 255.0 - torch.tensor(self.mean).float()) / torch.tensor(self.std).float()
            frames_sequence = frames_sequence.permute(0, 3, 1, 2)

        # Normalize the keypoint coordinates by dividing by the dimensions after applying the keypoint presence mask
        dims = torch.tensor([1280, 736]).float()
        cpts_sequence = cpts_sequence * cpts_presence_sequence / dims
        mask_invalid = (cpts_sequence[:, :, 0] < 0) | (cpts_sequence[:, :, 0] > 1) | (cpts_sequence[:, :, 1] < 0) | (cpts_sequence[:, :, 1] > 1)
        cpts_sequence[mask_invalid] = 0.

        return frames_sequence, mask_sequence, cpts_sequence, cpts_presence_sequence, frame_name_list