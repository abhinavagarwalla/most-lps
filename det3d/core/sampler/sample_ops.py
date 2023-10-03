import copy
import pathlib
import pickle
import time
from functools import partial, reduce

import numpy as np
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.utils.check import shape_mergeable


class DataBaseSamplerV2:
    def __init__(
        self,
        db_infos,
        groups,
        db_prepor=None,
        rate=1.0,
        global_rot_range=None,
        logger=None,
    ):
        for k, v in db_infos.items():
            logger.info(f"load {len(v)} {k} database infos")

        if db_prepor is not None:
            db_infos = db_prepor(db_infos)
            logger.info("After filter database:")
            for k, v in db_infos.items():
                logger.info(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self._use_group_sampling = False  # slower
        if any([len(g) > 1 for g in groups]):
            self._use_group_sampling = True
        if not self._use_group_sampling:
            self._group_db_infos = self.db_infos  # just use db_infos
            for group_info in groups:
                group_names = list(group_info.keys())
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
        else:
            for group_info in groups:
                group_dict = {}
                group_names = list(group_info.keys())
                group_name = ", ".join(group_names)
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
                self._group_name_to_names.append((group_name, group_names))
                # self._group_name_to_names[group_name] = group_names
                for name in group_names:
                    for item in db_infos[name]:
                        gid = item["group_id"]
                        if gid not in group_dict:
                            group_dict[gid] = [item]
                        else:
                            group_dict[gid] += [item]
                if group_name in self._group_db_infos:
                    raise ValueError("group must be unique")
                group_data = list(group_dict.values())
                self._group_db_infos[group_name] = group_data
                info_dict = {}
                if len(group_info) > 1:
                    for group in group_data:
                        names = [item["name"] for item in group]
                        names = sorted(names)
                        group_name = ", ".join(names)
                        if group_name in info_dict:
                            info_dict[group_name] += 1
                        else:
                            info_dict[group_name] = 1
                print(info_dict)

        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = prep.BatchSampler(v, k)
        self._enable_global_rot = False
        if global_rot_range is not None:
            if not isinstance(global_rot_range, (list, tuple, np.ndarray)):
                global_rot_range = [-global_rot_range, global_rot_range]
            else:
                assert shape_mergeable(global_rot_range, [2])
            if np.abs(global_rot_range[0] - global_rot_range[1]) >= 1e-3:
                self._enable_global_rot = True
        self._global_rot_range = global_rot_range

        self.label_to_idx_mapping = {'noise': 0, 'barrier': 1, 'bicycle': 2, 'bus': 3, 'car': 4, 'construction_vehicle': 5,
                             'motorcycle': 6, 'pedestrian': 7, 'traffic_cone': 8, 'trailer': 9, 'truck': 10, 
                             'driveable_surface': 11, 'other_flat': 12, 'sidewalk': 13, 'terrain': 14, 
                             'manmade': 15, 'vegetation': 16}
        
        # define for semantickitti
        # self.label_to_idx_mapping = { "unlabeled": 0, "car": 1, "bicycle": 2, "motorcycle": 3, "truck": 4, "other-vehicle": 5,
        #                     "person": 6, "bicyclist": 7, "motorcyclist": 8, "road": 9, "parking": 10, "sidewalk": 11, "other-ground": 12,
        #                     "building": 13, "fence": 14, "vegetation": 15, "trunk": 16, "terrain": 17, "pole": 18, "traffic-sign": 19}


    @property
    def use_group_sampling(self):
        return self._use_group_sampling

    def sample_all(
        self,
        root_path,
        gt_boxes,
        gt_names,
        num_point_features,
        random_crop=False,
        gt_group_ids=None,
        calib=None,
        road_planes=None,
    ):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
            self._sample_classes, self._sample_max_nums
        ):
            sampled_num = int(
                max_sample_num - np.sum([n == class_name for n in gt_names])
            )

            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled_groups = self._sample_classes
        if self._use_group_sampling:
            assert gt_group_ids is not None
            sampled_groups = []
            sample_num_per_class = []
            for group_name, class_names in self._group_name_to_names:
                sampled_nums_group = [sampled_num_dict[n] for n in class_names]
                sampled_num = np.max(sampled_nums_group)
                sample_num_per_class.append(sampled_num)
                sampled_groups.append(group_name)
            total_group_ids = gt_group_ids
        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, sampled_num in zip(sampled_groups, sample_num_per_class):
            if sampled_num > 0:
                if self._use_group_sampling:
                    sampled_cls = self.sample_group(
                        class_name, sampled_num, avoid_coll_boxes, total_group_ids
                    )
                else:
                    sampled_cls = self.sample_class_v2(
                        class_name, sampled_num, avoid_coll_boxes
                    )

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )

                    if sampled_gt_box.shape[1] == 8: #nuscenes
                        sampled_gt_box = np.concatenate((sampled_gt_box, np.zeros((sampled_gt_box.shape[0], 1))), axis=1) #adding in rotation as 0
                    if sampled_gt_box.shape[1] == 10: #semantickitti
                        sampled_gt_box = sampled_gt_box[:, [0,1,2,3,4,5,7,8,6]] #remove vz, bring r to last
                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0
                    )
                    if self._use_group_sampling:
                        if len(sampled_cls) == 1:
                            sampled_group_ids = np.array(sampled_cls[0]["group_id"])[
                                np.newaxis, ...
                            ]
                        else:
                            sampled_group_ids = np.stack(
                                [s["group_id"] for s in sampled_cls], axis=0
                            )
                        total_group_ids = np.concatenate(
                            [total_group_ids, sampled_group_ids], axis=0
                        )

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)

            num_sampled = len(sampled)
            s_points_list = []
            sem_labels_list = []
            base_index_list = []
            for info in sampled:
                try:
                    s_points = np.fromfile(
                        str(pathlib.Path(root_path) / info["path"]), dtype=np.float32
                    ).reshape(-1, num_point_features)

                    if "rot_transform" in info:
                        rot = info["rot_transform"]
                        s_points[:, :3] = box_np_ops.rotation_points_single_angle(
                            s_points[:, :4], rot, axis=2
                        )
                    s_points[:, :3] += info["box3d_lidar"][:3]
                    s_points_list.append(s_points)

                    sem_labels_list.append(np.ones((s_points[:,-1]==0).sum())*self.label_to_idx_mapping[info['name']])
                    base_index_mask = np.zeros(s_points.shape[0], dtype=np.bool)
                    base_index_mask[s_points[:,-1]==0] = True
                    base_index_list.append(base_index_mask)
                except Exception:
                    print(str(pathlib.Path(root_path) / info["path"]))
                    continue
            if random_crop:
                s_points_list_new = []
                sem_labels_list_new = []
                base_index_list_new = []
                assert calib is not None
                rect = calib["rect"]
                Trv2c = calib["Trv2c"]
                P2 = calib["P2"]
                gt_bboxes = box_np_ops.box3d_to_bbox(sampled_gt_boxes, rect, Trv2c, P2)
                crop_frustums = prep.random_crop_frustum(gt_bboxes, rect, Trv2c, P2)
                for i in range(crop_frustums.shape[0]):
                    s_points = s_points_list[i]
                    mask = prep.mask_points_in_corners(
                        s_points, crop_frustums[i : i + 1]
                    ).reshape(-1)
                    num_remove = np.sum(mask)
                    if num_remove > 0 and (s_points.shape[0] - num_remove) > 15:
                        s_points = s_points[np.logical_not(mask)]
                    s_points_list_new.append(s_points)
                    sem_labels_list_new.append(sem_labels_list[i][np.logical_not(mask)])
                    base_index_list_new.append(base_index_list_new[i][np.logical_not(mask)])
                s_points_list = s_points_list_new
                sem_labels_list = sem_labels_list_new
                base_index_list = base_index_list_new
            ret = {
                "gt_names": np.array([s["name"] for s in sampled]),
                "difficulty": np.array([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes,
                "points": np.concatenate(s_points_list, axis=0),
                "gt_masks": np.ones((num_sampled,), dtype=np.bool_),
                "sem_labels": np.concatenate(sem_labels_list, axis=0),
                "base_index_mask": np.concatenate(base_index_list, axis=0),
            }
            if self._use_group_sampling:
                ret["group_ids"] = np.array([s["group_id"] for s in sampled])
            else:
                ret["group_ids"] = np.arange(
                    gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled)
                )
        else:
            ret = None
        return ret

    def sample(self, name, num):
        if self._use_group_sampling:
            group_name = name
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(l) for l in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret),), dtype=np.int64)

    def sample_v1(self, name, num):
        if isinstance(name, (list, tuple)):
            group_name = ", ".join(name)
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(l) for l in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret),), dtype=np.int64)

    def sample_class_v2(self, name, num, gt_boxes):
        if name not in self._sampler_dict:
            name = name.replace('_', '-')
        sampled = self._sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        if sp_boxes.shape[1] == 8: #nuscenes
            sp_boxes = np.concatenate((sp_boxes, np.zeros((sp_boxes.shape[0], 1))), axis=1) #adding in rotation as 0
        # for semantickitt
        if sp_boxes.shape[1] == 10:
            sp_boxes = sp_boxes[:, [0,1,2,3,4,5,7,8,6]] #remove vx, bring r to last

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )

        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        if self._enable_global_rot:
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes, None, valid_mask, 0, 0, self._global_rot_range, num_try=100
            )

        sp_boxes_new = boxes[gt_boxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], np.zeros_like(sp_boxes_new[:, -1])
        )

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        coll_mat = prep.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                if self._enable_global_rot:
                    sampled[i - num_gt]["box3d_lidar"][:2] = boxes[i, :2]
                    sampled[i - num_gt]["box3d_lidar"][-1] = boxes[i, -1]
                    sampled[i - num_gt]["rot_transform"] = (
                        boxes[i, -1] - sp_boxes[i - num_gt, -1]
                    )
                valid_samples.append(sampled[i - num_gt])
        return valid_samples