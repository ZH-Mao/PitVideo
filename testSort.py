import pandas as pd

# 读取csv_file1
csv_file1 = pd.read_csv('/home/zhehua/codes/Pituitary-Segment-Centroid/data/Segmentation_Centroid_5_fold_validation_split_4structure/val5/image_centroid_fold5_train.csv')

# 将文件名按照视频ID和帧ID拆分成两列，并添加到列表的最后
csv_file1[['video_id', 'frame_id']] = csv_file1['image'].str.split('_', expand=True)
csv_file1['frame_id'] = csv_file1['frame_id'].str.split('.', expand=True)[0]  # 去除文件扩展名

# 将视频ID和帧ID转换为整数类型，以便排序
csv_file1['video_id'] = csv_file1['video_id'].astype(int)
csv_file1['frame_id'] = csv_file1['frame_id'].astype(int)

# 按视频ID和帧ID排序
csv_file1_sorted = csv_file1.sort_values(by=['video_id', 'frame_id']).reset_index(drop=True)

# 打印排序后的结果
print(csv_file1_sorted)
