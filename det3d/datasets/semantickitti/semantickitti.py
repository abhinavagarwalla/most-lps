import sys
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class SemanticKITTIDataset(PointCloudDataset):
  NumPointFeatures = 4 # x, y, z, intensity

  def __init__(
    self,
    info_path,
    root_path,
    cfg=None,
    pipeline=None,
    class_names=None,
    test_mode=False,
    sample=False,
    nsweeps=1,
    load_interval=1,
    **kwargs,
  ):
    self.load_interval = load_interval 
    self.sample = sample
    self.nsweeps = nsweeps
    print("Using {} sweeps".format(nsweeps))
    super(SemanticKITTIDataset, self).__init__(
      root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
    )

    self._info_path = info_path
    self._class_names = class_names
    self._num_point_features = SemanticKITTIDataset.NumPointFeatures if nsweeps == 1 else SemanticKITTIDataset.NumPointFeatures+1

  def reset(self):
    assert False 

  def load_infos(self, info_path):

    with open(self._info_path, "rb") as f:
      _semantickitti_infos_all = pickle.load(f)

    self._semantickitti_infos = _semantickitti_infos_all[::self.load_interval]

    print("Using {} Frames".format(len(self._semantickitti_infos)))

  def __len__(self):

    if not hasattr(self, "_semantickitti_infos"):
      self.load_infos(self._info_path)

    return len(self._semantickitti_infos)

  def get_sensor_data(self, idx):
    info = self._semantickitti_infos[idx]

    res = {
      "lidar": {
        "type": "lidar",
        "points": None,
        "annotations": None,
        "nsweeps": self.nsweeps, 
      },
      "metadata": {
        "image_prefix": self._root_path,
        "num_point_features": self._num_point_features,
        "token": info["token"],
      },
      "calib": None,
      "cam": {},
      "mode": "val" if self.test_mode else "train",
      "type": "SemanticKITTIDataset",
    }

    data, _ = self.pipeline(res, info)

    return data

  def __getitem__(self, idx):
    return self.get_sensor_data(idx)

  def evaluation(self, detections, output_dir=None, testset=False):
    from .semantickitti_common import _create_pd_detection, reorganize_info

    infos = self._semantickitti_infos 
    infos = reorganize_info(infos)

    _create_pd_detection(detections, infos, output_dir)

    print("Evaluation not implemented")

    return None, None