import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'  #GPU id
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import cv2

def read_segmentation_mask(file_path):
  return np.array(Image.open(file_path))/120

def get_confusion_matrix(pred_mask, true_mask, num_class):
  mask = (true_mask >= 0) & (true_mask < num_class)
  label = num_class * true_mask[mask].astype(int) + pred_mask[mask]
  count = np.bincount(label.astype(int), minlength=num_class**2)
  confusion_matrix = count.reshape(num_class, num_class)
  return confusion_matrix

def calculate_metrics(confusion_matrix):
  # 计算每个类别的TP, FP, FN
  tp = np.diag(confusion_matrix)
  fp = np.sum(confusion_matrix, axis=0) - tp
  fn = np.sum(confusion_matrix, axis=1) - tp

  # 计算IoU
  iou = tp / (tp + fp + fn)

  # 计算Precision
  precision = tp / (tp + fp)

  # 计算Recall
  recall = tp / (tp + fn)

  # 计算F1 score
  f1 = 2 * (precision * recall) / (precision + recall)

  return iou, precision, recall, f1

def calculate_keypoint_distance(pred_x, pred_y, true_x, true_y):
  if pred_x == -100 or pred_y == -100 or true_x == -100 or true_y == -100:
      return None
  return np.sqrt(((pred_x - true_x)*1280)**2 + ((pred_y - true_y)*720)**2)

def visualize_results(image_path, pred_mask, true_mask, pred_landmarks, true_landmarks, save_path):
  # Read the original image
  ori_image = np.array(Image.open(image_path))
  result_image = ori_image.copy()

  # Define colors
  transparent = (0, 0, 0, 0)
  cls_colors = [(77, 77, 255, 200), (255, 255, 77, 200), 
                (180, 77, 224, 255), (77, 255, 77, 255),
                (122, 233, 222, 255), (255, 77, 255, 255)]

  # # Overlay ground truth mask contours
  # for class_id in [1, 2]:
  #     mask = (true_mask == class_id).astype(np.uint8)
  #     _, binary = cv2.threshold(mask, 0.5, 255, 0)
  #     contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #     cv2.drawContours(result_image, contours, -1, cls_colors[class_id-1][:3], 3)
  
  # # Overlay predicted mask contours
  # for class_id in [1, 2]:
  #     mask = (pred_mask == class_id).astype(np.uint8)
  #     _, binary = cv2.threshold(mask, 0.5, 255, 0)
  #     contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #     cv2.drawContours(result_image, contours, -1, cls_colors[class_id-1][:3], 3)

  # Convert from numpy to PIL
  result_image = Image.fromarray(np.uint8(result_image))
  draw = ImageDraw.Draw(result_image)

  # Overlay ground truth masks, comment out if no need
  mask_image = Image.new("RGBA", result_image.size, transparent)
  for class_id in [1, 2]:
      mask = (true_mask == class_id).astype(np.uint8) * 200
      mask_image.paste(cls_colors[class_id-1], (0, 0), Image.fromarray(mask).convert("L"))
  result_image.paste(mask_image, (0, 0), mask_image)


  # # Overlay predicted masks
  # for class_id in [1, 2]:
  #     cls_mask = Image.new("RGBA", result_image.size, transparent)
  #     mask = (pred_mask == class_id).astype(np.uint8) * 200
  #     cls_mask.paste(cls_colors[class_id-1], (0, 0), Image.fromarray(mask).convert("L"))
  #     result_image.paste(cls_mask, (0, 0), cls_mask)

  # Overlay ground truth landmarks as circles
  for j, landmark in enumerate(true_landmarks):
      x, y = landmark
      if x != -100 and y != -100:
          draw.ellipse([(x*1280 - 20, y*720 - 20), (x*1280 + 20, y*720 + 20)], fill=cls_colors[j+2])

  # # Overlay predicted landmarks as crosses
  # for j, landmark in enumerate(pred_landmarks):
  #     x, y = landmark
  #     if x != -100 and y != -100:
  #         draw.line([(x*1280 - 15, y*720 - 15), (x*1280 + 15, y*720 + 15)], fill=cls_colors[j+2], width=6)
  #         draw.line([(x*1280 + 15, y*720 - 15), (x*1280 - 15, y*720 + 15)], fill=cls_colors[j+2], width=6)

  # Save the result
  result_image.save(save_path)

def evaluate_model(pred_seg_folder, true_seg_folder, pred_keypoints_file, true_keypoints_file, image_folder, output_folder):
  # 读取关键点数据
  pred_df = pd.read_excel(pred_keypoints_file)
  true_df = pd.read_excel(true_keypoints_file)
  
  # 初始化混淆矩阵和关键点距离列表
  confusion_matrix = np.zeros((3, 3))
  keypoint_distances = []
  
  for _, row in pred_df.iterrows():
      image_name = row['image']
      
      # 读取语义分割掩码
      pred_mask = read_segmentation_mask(os.path.join(pred_seg_folder, image_name))
      true_mask = read_segmentation_mask(os.path.join(true_seg_folder, image_name))
      
      # 更新混淆矩阵
      confusion_matrix += get_confusion_matrix(pred_mask, true_mask, 3)
      
      # 计算关键点距离
      true_row = true_df[true_df['image'] == image_name].iloc[0]
      pred_landmarks = []
      true_landmarks = []
      for i in [3, 5, 8, 10]:
          pred_x, pred_y = row[f'structure_{i}_x'], row[f'structure_{i}_y']
          true_x, true_y = true_row[f'structure_{i}_x'], true_row[f'structure_{i}_y']
          pred_landmarks.append((pred_x, pred_y))
          true_landmarks.append((true_x, true_y))
          distance = calculate_keypoint_distance(pred_x, pred_y, true_x, true_y)
          if distance is not None:
              keypoint_distances.append(distance)
      
      # 可视化结果
      image_path = os.path.join(image_folder, image_name)
      save_path = os.path.join(output_folder, f"pseudo_{image_name}")
      visualize_results(image_path, pred_mask, true_mask, pred_landmarks, true_landmarks, save_path)
  
  # 计算各项指标
  iou, precision, recall, f1 = calculate_metrics(confusion_matrix)
  
  # 计算平均关键点距离
  avg_keypoint_distance = np.mean(keypoint_distances)
  mpck = np.zeros(4)
  mpck[0] = (np.array(keypoint_distances)<36).astype(int).sum()/len(keypoint_distances)*100
  mpck[1] = (np.array(keypoint_distances)<72).astype(int).sum()/len(keypoint_distances)*100
  mpck[2] = (np.array(keypoint_distances)<108).astype(int).sum()/len(keypoint_distances)*100
  mpck[3] = (np.array(keypoint_distances)<144).astype(int).sum()/len(keypoint_distances)*100
  
  
  return iou, precision, avg_keypoint_distance, mpck, recall, f1

if __name__ == "__main__":
  # # option 1: evaluate pseudo label with true label
  # # true label is manual annotation
  # # pred label is pseudo label got by co-tracker
  # # use pred label to index true label since the numbder of pseudo label is less than true label, template is not included in pseudo label
  # true_seg_folder = '/home/zhehua/data/PitDatasets/Annotation_mask_sella_clival_recess'
  # pred_seg_folder = '/home/zhehua/data/PitDatasets/pseudo_Annotation_mask_sella_clival_recess_for_pseudo_label_eval'
  # true_keypoints_file = '/home/zhehua/data/PitDatasets/Annotation_centroid_4structure_V3_final_reorganized.xlsx'
  # pred_keypoints_file = '/home/zhehua/data/PitDatasets/pseudo_Annotation_centroid_4structure_for_pseudo_label_eval.xlsx'
  
  # # option 2: plot true label only
  # true_seg_folder = '/home/zhehua/data/PitDatasets/Annotation_mask_sella_clival_recess'
  # pred_seg_folder = true_seg_folder
  # true_keypoints_file = '/home/zhehua/data/PitDatasets/Annotation_centroid_4structure_V3_final_reorganized.xlsx'
  # pred_keypoints_file = true_keypoints_file
  
  
  # option 3: plot pseudo label only
  true_seg_folder = '/home/zhehua/data/PitDatasets/pseudo_Annotation_mask_sella_clival_recess_plus_extra'
  pred_seg_folder = true_seg_folder
  true_keypoints_file = '/home/zhehua/data/PitDatasets/data_splitting/fold1_makevideo/pseudo_Annotation_centroid_4structure_plus_extra_make_video.xlsx'
  pred_keypoints_file = true_keypoints_file

  image_folder = '/home/zhehua/data/PitDatasets/Extracted_video_frames_5FPS'  
  output_folder = '/home/zhehua/data/Results/pituitary/groudtruth_pseudo_visualization_makevideo'  

  if not os.path.exists(output_folder):
      os.makedirs(output_folder)

  iou, precision, avg_keypoint_distance, mpck, recall, f1 = evaluate_model(
      pred_seg_folder, true_seg_folder, pred_keypoints_file, true_keypoints_file, image_folder, output_folder
  )

  print("IoU for each class:", iou)
  print("Precision for each class:", precision)
  print("Recall for each class:", recall)
  print("F1 score for each class:", f1)
  print("Average Keypoint Distance:", avg_keypoint_distance)
  print("MPCK:", mpck)
