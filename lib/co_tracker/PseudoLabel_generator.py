import os
os.environ["CUDA_VISIBLE_DEVICES"]='0' 
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import DBSCAN

import torch

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor
from shapely import MultiPoint, Polygon, concave_hull


def get_video_from_frames(annotation_list, video_frames, idx, image_root, video_length):
    labeled_frame_name = annotation_list.iloc[idx, 1]        
    # find index of labeled_frame_name in video sequence
    matching_rows = video_frames.loc[video_frames['image'] == labeled_frame_name]
    frameID_in_video = matching_rows.index.tolist()
    
    frames = []
    frames_name = []
    for i in range(video_length-1, -1, -1):
        # read image
        image_name = video_frames.iloc[frameID_in_video[0]-i, 0]
        image_path = os.path.join(image_root, image_name)
        image = Image.open(image_path)
        image = np.array(image)
        frames.append(image)
        frames_name.append(image_name)
    
    return np.stack(frames), frames_name

def get_cpts_from_annotation(annotation_list, idx):
    # centroid points coordinates (cpts)
    cpts_ref_norm = annotation_list.iloc[idx, 2:].values
    cpts_ref_norm = cpts_ref_norm.astype('float').reshape(-1, 2)
    # cpts_presence = np.float32(cpts != -100)
    cpts_presence = (cpts_ref_norm != -100)
    cpts = cpts_ref_norm*cpts_presence
    cpts[:, 0] = cpts[:, 0]*1280
    cpts[:, 1] = cpts[:, 1]*720
    return cpts_ref_norm, cpts, cpts_presence[:,1]

def get_mask_from_frames(annotation_list, idx, mask_root):
    fold = annotation_list.iloc[idx, 0]
    labeled_frame_name = annotation_list.iloc[idx, 1] 
    mask_path = os.path.join(mask_root, labeled_frame_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)      
    return mask, fold


def traking_points_to_mask(visible_points_i, mask_w, mask_h, eps=30, min_samples=10):
    
    # create blank mask image
    img_size = (mask_w, mask_h)
    mask_img = Image.new('L', img_size, 0)
    draw = ImageDraw.Draw(mask_img)
    if len(visible_points_i) > 2:
        # use DBSCAN to automatically group the points, eps and min_samples are adjusted according to the dataset
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(visible_points_i)
        labels = db.labels_

        # generate mask for each group (excluding noise data)
        for label in set(labels):
            if label == -1:
                # -1 represents noise points, skip
                continue
            points_in_cluster = visible_points_i[labels == label]

            # generate convex or concave hull
            hull = concave_hull(MultiPoint(points_in_cluster), ratio=0.4)
            
            # draw polygon
            hull_points = list(zip(*hull.exterior.coords.xy))
            if len(hull_points) > 2:
                draw.polygon(hull_points, fill=255)

    # For debug
    # mask_img.save('mask.png')
    return np.array(mask_img)
    
if __name__ == '__main__':
    root = '/home/zhehua/data/PitDatasets/'
    image_path = 'Images_5FPS/'
    mask_path = 'Masks/'
    annotation_list_path = 'Landmarks_2_structure_4.csv'
    video_index_path = 'videoIndex_5fps.csv'
    output_path = 'Masks_pseudo4/'
    output_csv = "Landmarks_pseudo_structure_4_2.csv"  # replace with the path to save the CSV file
    
    video_length = 5
    class_values = [1,2]
    
    image_root = os.path.join(root, image_path)
    mask_root = os.path.join(root, mask_path)
    annotation_list = pd.read_csv(os.path.join(root, annotation_list_path))
    video_frames = pd.read_csv(os.path.join(root, video_index_path))
    
    header_df = pd.read_csv(os.path.join(root, annotation_list_path), nrows=0)  # read header
    header = header_df.columns.tolist()
    # prepare data
    data = []
    
    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            './checkpoints/cotracker2.pth'
        )
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for idx in range(len(annotation_list.index)):
        video, frames_name = get_video_from_frames(annotation_list, video_frames, idx, image_root, video_length)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        cpts_ref_norm, cpts_ref, cpts_presence_ref  = get_cpts_from_annotation(annotation_list, idx)
        mask_ref, fold = get_mask_from_frames(annotation_list, idx, mask_root)
        height, width = mask_ref.shape
               
        video =  video.to(device)
        ####################################################################################################
        ## Section1: mask label generator
        if not os.path.exists(os.path.join(root,output_path)):
            os.makedirs(os.path.join(root,output_path))
        mask2 = []
        for v in class_values:
            mask1 = []
            mask = (mask_ref == v).astype(np.uint8)*255
            if cv2.countNonZero(mask) > 0:
                pred_tracks, pred_visibility = model(video, grid_size=100, segm_mask=torch.from_numpy(mask)[None, None], grid_query_frame=video_length-1, backward_tracking=True)

                # visulization
                vis = Visualizer(
                    save_dir='./videos',
                    pad_value=100,
                    linewidth=2,
                )
                vis.visualize(
                    video=video,
                    tracks=pred_tracks,
                    visibility=pred_visibility,
                    filename=frames_name[-1].split('.')[0])

                for i in range(video_length-1):
                    pred_tracks_i = pred_tracks[:,i,:,:]
                    pred_visibility_i = pred_visibility[:,i,:]
                    
                    visible_points_i = pred_tracks_i[pred_visibility_i].cpu().numpy()
                    mask_img = traking_points_to_mask(visible_points_i, width, height)
                    mask1.append(mask_img)
            else:
                for i in range(video_length-1):
                    # create mask image
                    img_size = (width, height)
                    mask_img = Image.new('L', img_size, 0)
                    mask1.append(mask_img)
            mask1.append(mask)
            mask2.append(mask1)
        
        final_masks = []
        for i in range(video_length):
            # create mask image
            img_size = (width, height)
            final_mask = Image.new('L', img_size, 0)
            final_mask = np.array(final_mask)
            for j in range(len(mask2)):
                final_mask[mask2[j][i]==255]=class_values[j]            
            # add the final mask to the list
            final_masks.append(final_mask)
            
        # save the final mask as an image
        for frame_name, mask in zip(frames_name, final_masks):
            mask_image = Image.fromarray((mask * 127).astype(np.uint8))  # normalize to the range of 0-255, for easy viewing
            # frame_name = frame_name.replace('.png', '_mask.png')  # assume the original image is .png format, modify according to the实际情况
            mask_image.save(os.path.join(root, output_path,frame_name))

        ####################################################################################################
        ## Section2: Landmark generator
        # get indices of points labeled
        """
        Explain:
        Assume the points need to be tracked are
        a = [[-100, -100],
            [915.10504064, 280.38594312],
            [483.10263168, 231.82853904],
            [403.6611648 , 391.9270356 ]]
        cpts_presence_ref = [False, True, True, True]
        presence_indices = [1,2,3]
        
        queries = [[915.10504064, 280.38594312],
            [483.10263168, 231.82853904],
            [403.6611648 , 391.9270356 ]]

        Assume:
        pred_tracks = [[915.10504064, 280.38594312],
            [483.10263168, 231.82853904],
            [403.6611648 , 391.9270356 ]]
            
        pred_visibility = [False, True, True]
        
        visible_points_indices_i = [1, 2]
        cpts_tracked_indices = presence_indices[visible_points_indices_i] = [2,3]
        """
        presence_indices = np.where(cpts_presence_ref)[0]
        if not presence_indices.size == 0:
            queries = cpts_ref[cpts_presence_ref]
            # frame id of the reference in the video clip, a clip consisting 5 frames, frame_id=4,
            frame_id = np.ones(len(queries))*(video_length-1)
            queries = np.insert(queries, 0, frame_id, axis=1)
            queries = torch.from_numpy(queries).float() # add .float(), because numpy generates small numbers as double by default, but the model expects FloatTensor
            queries = queries.to(device)
            pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)
            # visulization
            vis = Visualizer(
                save_dir='./videos',
                pad_value=100,
                linewidth=2,
            )
            vis.visualize(
                video=video,
                tracks=pred_tracks,
                visibility=pred_visibility,
                filename=frames_name[-1].split('.')[0]+'_landmark')
            
            pred_visibility = pred_visibility.cpu().numpy()
            pred_tracks = pred_tracks.cpu().numpy()            
            cpts_tracked = []
            for i in range(video_length-1):
                cpts_tracked_i = np.ones((len(cpts_presence_ref), 2))*(-100)
                pred_tracks_i = pred_tracks[:,i, :, :].squeeze(0) # squeeze() specify the dimension, otherwise there will be a Bug if there is only one coordinate
                visible_points_indices_i = np.where(pred_visibility[:,i,:].squeeze())[0]
                if not visible_points_indices_i.size ==0:
                    cpts_tracked_indices = presence_indices[visible_points_indices_i]
                    cpts_tracked_i[cpts_tracked_indices] = pred_tracks_i[visible_points_indices_i]/[1280, 720]
                    
                    # # the distance between adjacent frames should not be too far, set a threshold to exclude error tracking points (actually tested, no such points appear) 
                    # distance=np.sqrt(np.sum((pred_tracks_i[visible_points_indices_i] - cpts_ref[cpts_tracked_indices])**2, axis=1))
                    # thresh = np.where(distance < 50)[0]
                    # if not thresh.size ==0:
                    #     cpts_tracked_i[cpts_tracked_indices[thresh]] = pred_tracks_i[visible_points_indices_i[thresh]]/[1280, 720]                    
                cpts_tracked.append(cpts_tracked_i)
            cpts_tracked.append(cpts_ref_norm)
        else:
            cpts_tracked_i = np.ones((len(cpts_presence_ref), 2))*(-100)
            cpts_tracked = []
            for i in range(video_length):
                cpts_tracked.append(cpts_tracked_i)
        
        for i, frame_name in enumerate(frames_name):
            row = [fold, frame_name]
            for coord in cpts_tracked[i]:
                row.extend(coord)  # add coordinate points
            data.append(row)

    # create DataFrame
    df = pd.DataFrame(data, columns=header)
    # save to CSV
    df.to_csv(os.path.join(root, output_csv), index=False)