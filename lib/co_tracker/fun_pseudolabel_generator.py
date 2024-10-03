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


def get_video_from_frames(frame_names, image_root, video_length, size):
    # labeled_frame_name = annotation_list.iloc[idx, 1]        
    # find index of labeled_frame_name in video sequence
    # matching_rows = video_frames.loc[video_frames['image'] == labeled_frame_name]
    # frameID_in_video = matching_rows.index.tolist()
    
    frames = []
    frames_name = []
    for i in range(video_length):
        # read image
        image_name = frame_names[i][0]
        image_path = os.path.join(image_root, image_name)
        image = Image.open(image_path)
        image = image.resize((size[-1], size[-2]), Image.BILINEAR)
        image = np.array(image)
        frames.append(image)
        frames_name.append(image_name)
    
    return np.stack(frames), frames_name

def get_video_from_frames(last_frame_name, video_frames, image_root, video_length):
    '''last_frame_name is the name of the mask in the mini_batch    
    '''
    labeled_frame_name = last_frame_name      
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

# def traking_points_to_mask(visible_points_i, mask_w, mask_h):
#     hull = concave_hull(MultiPoint(visible_points_i), ratio=0.4)
#     # 创建mask图片
#     img_size = (mask_w, mask_h)
#     mask_img = Image.new('L', img_size, 0)
#     draw = ImageDraw.Draw(mask_img)
#     hull_points = list(zip(*hull.exterior.coords.xy))
#     if len(hull_points) > 2:
#         draw.polygon(hull_points, fill=255)
    
#     # For debug
#     # mask_img.save('mask.png')
#     return np.array(mask_img)

def traking_points_to_mask(visible_points_i, mask_w, mask_h, eps=30, min_samples=10):
    
    # 创建空白mask图片
    img_size = (mask_w, mask_h)
    mask_img = Image.new('L', img_size, 0)
    draw = ImageDraw.Draw(mask_img)
    if len(visible_points_i) > 2:
        # 使用DBSCAN自动分组，eps和min_samples根据数据集调整
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(visible_points_i)
        labels = db.labels_

        # 对每个组(排除噪声数据)生成mask
        for label in set(labels):
            if label == -1:
                # -1表示噪声点，跳过
                continue
            points_in_cluster = visible_points_i[labels == label]

            # 生成凸包或凹包
            hull = concave_hull(MultiPoint(points_in_cluster), ratio=0.4)
            
            # 绘制多边形
            hull_points = list(zip(*hull.exterior.coords.xy))
            if len(hull_points) > 2:
                draw.polygon(hull_points, fill=255)

    # For debug
    # mask_img.save('mask.png')
    return np.array(mask_img)

def traking_points_to_mask_dilated_points(visible_points_i, mask_w, mask_h, dilated_mask=3):
    
    # 创建空白mask图片
    mask_img = np.zeros((mask_h, mask_w), dtype=np.uint8)
    int_coords = visible_points_i.astype(int)
    mask_img[int_coords[:, 1], int_coords[:, 0]] = 255 
    
    # 定义结构元素（kernel）用于膨胀，尺寸为connectivity_threshold
    kernel_size = dilated_mask * 2 + 1  # 确保中心点到边界的距离为connectivity_threshold
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 膨胀操作
    dilated_mask = cv2.dilate(mask_img, kernel, iterations=1)
    # For debug
    cv2.imwrite('output_image.jpg', dilated_mask)
    return dilated_mask
    
def predict_propagation(cfg, frame_names, mask, cpts_ref_norm, cpts_presence_ref, mask_id):
    cpts_presence_ref = (cpts_presence_ref != 0)[:,1].cpu().numpy()
    root = '/home/zhehua/data/PitDatasets/'
    image_path = 'Extracted_video_frames_5FPS/'
    mask_path = 'Annotation_mask_sella_clival_recess/'
    annotation_list_path = 'Annotation_centroid_4structure_V3_final_reorganized.xlsx'
    video_index_path = 'Extracted_video_frames_5FPS.xlsx'
    output_path = 'pseudo_Annotation_mask_sella_clival_recess_v4/'
    output_csv = "pseudo_Annotation_centroid_4structure_v4.xlsx"  # 替换为你要保存的CSV文件的路径
    
    video_length = 10
    # class_values = [1,2]
    mask = torch.argmax(mask, dim=0).cpu().numpy()
    size = mask.shape
    class_values = np.unique(mask).astype('int32').tolist()[1:]
    
    image_root = os.path.join(root, image_path)
    # mask_root = os.path.join(root, mask_path)
    # annotation_list = pd.read_csv(os.path.join(root, annotation_list_path))
    video_frames = pd.read_excel(os.path.join(root, video_index_path))
    
    header_df = pd.read_excel(os.path.join(root, annotation_list_path), nrows=0)  # 读取表头
    header = header_df.columns.tolist()
    # 准备数据
    data = []
    
    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            '/home/zhehua/codes/PitVideo-Segment-Landmark/lib/co_tracker/checkpoints/cotracker2.pth'
        )
    )
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # for idx in range(len(frame_names)):
    video, frames_name = get_video_from_frames(frame_names[-1], video_frames, image_root, video_length)
    mask_indeces_in_video = [frames_name.index(element) for element in frame_names]
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    cpts_ref = (cpts_ref_norm*torch.tensor([size[-1], size[-2]], dtype=cpts_ref_norm.dtype, device=cpts_ref_norm.device)).cpu().numpy()
    mask_ref = mask
    height, width = mask_ref.shape
            
    video =  video.to(device)
    ####################################################################################################
    ## Section1: mask label generator
    # if not os.path.exists(os.path.join(root,output_path)):
    #     os.makedirs(os.path.join(root,output_path))
    mask2 = []
    
    # Track all non-zero points within the mask
    for v in class_values:
        mask1 = []
        mask = (mask_ref == v).astype(np.uint8)*255
        if cv2.countNonZero(mask) > 0:
            pred_tracks, pred_visibility = model(video, grid_size=400, segm_mask=torch.from_numpy(mask)[None, None], grid_query_frame=mask_indeces_in_video[mask_id], backward_tracking=True)
            # pred_tracks, pred_visibility = model(video, segm_mask=torch.from_numpy(mask)[None, None], grid_query_frame=video_length-temp_length+mask_id, backward_tracking=True)

            # # visulization
            # vis = Visualizer(
            #     save_dir='./videos',
            #     pad_value=100,
            #     linewidth=2,
            # )
            # vis.visualize(
            #     video=video,
            #     tracks=pred_tracks,
            #     visibility=pred_visibility,
            #     filename=frames_name[-1].split('.')[0])

            for i in range(video_length):
                if i != mask_indeces_in_video[mask_id]:
                    pred_tracks_i = pred_tracks[:,i,:,:]
                    pred_visibility_i = pred_visibility[:,i,:]
                    
                    visible_points_i = pred_tracks_i[pred_visibility_i].cpu().numpy()
                    mask_img = traking_points_to_mask_dilated_points(visible_points_i, width, height, dilated_mask=3)
                    mask1.append(mask_img)
                else:
                    mask1.append(mask)
        else:
            for i in range(video_length):
                # 创建mask图片
                img_size = (width, height)
                mask_img = Image.new('L', img_size, 0)
                mask1.append(mask_img)
        mask2.append(mask1)
    
    final_masks = []
    for i in range(video_length):
        # 创建mask图片
        img_size = (width, height)
        final_mask = Image.new('L', img_size, 0)
        final_mask = np.array(final_mask)
        for j in range(len(mask2)):
            final_mask[mask2[j][i]==255]=class_values[j]            
        # 将最终的mask添加到列表中
        final_masks.append(final_mask)
        
    # # 保存最终的mask为图片
    # for frame_name, mask in zip(frames_name, final_masks):
    #     mask_image = Image.fromarray((mask * 120).astype(np.uint8))  # 标准化到0-255范围内，方便观看
    #     # frame_name = frame_name.replace('.png', '_mask.png')  # 假设原图是.png格式，根据实际情况修改
    #     mask_image.save(os.path.join(root, output_path,frame_name))

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
        # frame_id = np.ones(len(queries))*(video_length-1)
        frame_id = np.ones(len(queries))*(mask_indeces_in_video[mask_id])
        queries = np.insert(queries, 0, frame_id, axis=1)
        queries = torch.from_numpy(queries).float() # 要加上.float(), 因为numpy生成小数默认为double, 但是模型期望FloatTensor
        queries = queries.to(device)
        pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)
        # # visulization
        # vis = Visualizer(
        #     save_dir='./videos',
        #     pad_value=100,
        #     linewidth=2,
        # )
        # vis.visualize(
        #     video=video,
        #     tracks=pred_tracks,
        #     visibility=pred_visibility,
        #     filename=frames_name[-1].split('.')[0]+'_landmark')
        
        pred_visibility = pred_visibility.cpu().numpy()
        pred_tracks = pred_tracks.cpu().numpy()            
        cpts_tracked = []
        for i in range(video_length):
            if i != mask_indeces_in_video[mask_id]:
                cpts_tracked_i = np.ones((len(cpts_presence_ref), 2))*(-100)
                pred_tracks_i = pred_tracks[:,i, :, :].squeeze(0) # squeeze()要指定维度，不然只有一个坐标的话会出Bug
                visible_points_indices_i = np.where(pred_visibility[:,i,:].squeeze())[0]
                if not visible_points_indices_i.size ==0:
                    cpts_tracked_indices = presence_indices[visible_points_indices_i]
                    cpts_tracked_i[cpts_tracked_indices] = pred_tracks_i[visible_points_indices_i]/[size[-1], size[-2]]                                  
                cpts_tracked.append(cpts_tracked_i)
            else:
                cpts_tracked.append(cpts_ref_norm)
    else:
        cpts_tracked_i = np.ones((len(cpts_presence_ref), 2))*(-100)
        cpts_tracked = []
        for i in range(video_length):
            cpts_tracked.append(cpts_tracked_i)
            
    return final_masks[mask_indeces_in_video[mask_id+1]], cpts_tracked[mask_indeces_in_video[mask_id+1]]
    
#     for i, frame_name in enumerate(frames_name):
#         row = [fold, frame_name]
#         for coord in cpts_tracked[i]:
#             row.extend(coord)  # 添加坐标点
#         data.append(row)

# # 创建DataFrame
# df = pd.DataFrame(data, columns=header)
# # 保存到CSV
# df.to_csv(os.path.join(root, output_csv), index=False)