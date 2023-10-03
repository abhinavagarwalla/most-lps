import os.path as osp
import numpy as np
import pickle
import random

from pathlib import Path
from functools import reduce
from typing import Tuple, List
import os 
import json 
from tqdm import tqdm
import argparse

try:
  import tensorflow as tf
  tf.enable_eager_execution()
except:
  print("No Tensorflow")

from .semantickitti_base import SemanticKITTI


def sort_frame(frames):
  indices = [] 

  for f in frames:
    seq_id = int(f.split("_")[1])
    frame_id= int(f.split("_")[3][:-4])

    idx = seq_id * 1000 + frame_id
    indices.append(idx)

  rank = list(np.argsort(np.array(indices)))

  frames = [frames[r] for r in rank]
  return frames

def get_available_frames(root, split):
  dir_path = os.path.join(root, split, 'lidar')
  available_frames = list(os.listdir(dir_path))

  sorted_frames = sort_frame(available_frames)

  print(split, " split ", "exist frame num:", len(available_frames))
  return sorted_frames


def create_semantickitti_infos(root_path, split='train', nsweeps=1):
  dataset = SemanticKITTI(root_path, return_reflectance=True)
  dataset_split = dataset.get_split(split)
  semantic_kitti_infos = []

  print("Do think about converting to KITTI format (see Waymo)")

  for idx in tqdm(range(len(dataset_split))):
      print(f"Processing id: {idx}")

      data = dataset_split.get_data(idx, return_boxes=True)
      pc_path = str(dataset_split.path_list[idx])
      label_path = pc_path.replace('velodyne', 'bboxes').replace('.bin', '.boxes')
      sem_label_path = pc_path.replace('velodyne', 'labels').replace('.bin', '.label')

      info = {
          "path": pc_path,
          "anno_path": label_path,
          "sem_label_path": sem_label_path,
          "token": idx,
          "timestamp": None,
          "sweeps": [] #for aggregating, presently no need
      }

      if split != 'test':
          gt_boxes = np.array([bbox.to_xyzwhlr() for bbox in data['bounding_boxes']]).reshape(-1, 7)
          gt_names = np.array([dataset.label_to_names[bbox.label_class] for bbox in data['bounding_boxes']])

          info['gt_boxes'] = gt_boxes
          info['gt_names'] = gt_names

      semantic_kitti_infos.append(info)

  print(f"length of info: {len(semantic_kitti_infos)}")

  import pickle
  with open(os.path.join(root_path, f"infos_{split}_single_sweep.pkl"), 'wb') as f:
      pickle.dump(semantic_kitti_infos, f)


def parse_args():
  parser = argparse.ArgumentParser(description="Waymo 3D Extractor")
  parser.add_argument("--path", type=str, default="data/Waymo/tfrecord_training")
  parser.add_argument("--info_path", type=str)
  parser.add_argument("--result_path", type=str)
  parser.add_argument("--gt", action='store_true' )
  parser.add_argument("--tracking", action='store_true')
  args = parser.parse_args()
  return args


def reorganize_info(infos):
  new_info = {}

  for info in infos:
    token = info['token']
    new_info[token] = info

  return new_info 

if __name__ == "__main__":
  args = parse_args()

  with open(args.info_path, 'rb') as f:
    infos = pickle.load(f)
  
  if args.gt:
    _create_gt_detection(infos, tracking=args.tracking)
    exit() 

  infos = reorganize_info(infos)
  with open(args.path, 'rb') as f:
    preds = pickle.load(f)
  _create_pd_detection(preds, infos, args.result_path, tracking=args.tracking)
