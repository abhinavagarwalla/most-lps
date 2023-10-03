import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os

from nuscenes.nuscenes import NuScenes
# from det3d.datasets.nuscenes.nuscenes import NuScenesDataset

from ..registry import PIPELINES

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, virtual=False):
    if virtual:
        # WARNING: hard coded for nuScenes 
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        tokens = path.split('/')
        seg_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        data_dict = np.load(seg_path, allow_pickle=True).item()

        # remove reflectance as other virtual points don't have this value  
        virtual_points1 = data_dict['real_points'][:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] 
        virtual_points2 = data_dict['virtual_points']

        points = np.concatenate([points, np.ones([points.shape[0], 15-num_point_feature])], axis=1)
        virtual_points1 = np.concatenate([virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1)
        virtual_points2 = np.concatenate([virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1)
        points = np.concatenate([points, virtual_points1, virtual_points2], axis=0).astype(np.float32)
    else:
        # points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        points = np.fromfile(path.replace('/project_data/ramanan/aa4/', '/data/'), dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points

def read_file_kitti(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        raise NotImplementedError
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :num_point_feature]

    return points

def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, virtual=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), virtual=virtual).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), virtual=res["virtual"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]
            base_index_list = [np.ones(points.shape[0], dtype=bool)]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in range(len(info["sweeps"])):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)
                base_index_list.append(np.zeros(points_sweep.shape[0], dtype=bool))
            
            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
            base_index_mask = np.concatenate(base_index_list, axis=0)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
            res["lidar"]["base_index_mask"] = base_index_mask
        
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        elif self.type == "SemanticKITTIDataset":
            nsweeps = res["lidar"]["nsweeps"]
            lidar_path = Path(info["path"])
            points = read_file_kitti(str(lidar_path))

            times = np.zeros((points.shape[0], 2))
            res["lidar"]["points"] = points

        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        version = kwargs.get("version", 'v1.0-mini')
        self.dataroot = kwargs.get("dataroot", "/home/devenish/Desktop/capstone/datasets/nuscenes/v1.0-mini/")
        self.nusc = NuScenes(version=version, dataroot=self.dataroot)

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            
            #append rotation = 0 since certain augmentation assume last token as rotation
            if gt_boxes.shape[1] == 8:
                gt_boxes = np.concatenate([gt_boxes, np.zeros((gt_boxes.shape[0], 1))], axis=1)
                
            sample = self.nusc.get('sample', info['token'])
            pano_file = self.nusc.get('panoptic', sample['data']['LIDAR_TOP'])['filename']
            inst_label_list = [np.load(os.path.join(self.dataroot, pano_file))['data']]

            inst_label_all = np.concatenate(inst_label_list, axis=0)

            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
                "panoptic_label": inst_label_all,
                # "base_index_mask": base_index_mask
            }
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        elif res["type"] == "SemanticKITTIDataset" and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes = gt_boxes[:, [0,1,2,3,4,5,7,8,6]] #converting boxes to: x,y,z,w,l,h,vx,vy,r
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"]
            }
        else:
            pass 

        return res, info
