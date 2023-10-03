import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES

import numba as nb

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)
        self.learning_map = cfg.get('learning_map', {1: 0,5: 0,7: 0,8: 0,10: 0,11: 0,13: 0,19: 0,20: 0,0: 0,29: 0,31: 0,9: 1,14: 2,15: 3,16: 3,17: 4,18: 5,21: 6,2: 7,3: 7,4: 7,6: 7,12: 8,22: 9,23: 10,24: 11,25: 12,26: 13,27: 14,28: 15,30: 16})

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset", "SemanticKITTIDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
            if self.mode == "train":
                sem_labels = np.vectorize(self.learning_map.__getitem__)(res["lidar"]["annotations"]["panoptic_label"] // 1000).astype(np.uint32)
            base_index_mask = res["lidar"]["base_index_mask"]
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    sampled_sem_labels = sampled_dict["sem_labels"]
                    sampled_base_indices = sampled_dict["base_index_mask"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    if (sampled_points[:,-1]==0).sum() != sampled_sem_labels.shape[0]:
                        assert((sampled_points[:,-1]==0).sum() == sampled_sem_labels.shape[0])
                        assert((points[:,-1]==0).sum() == sem_labels.shape[0])
                    points = np.concatenate([sampled_points, points], axis=0)
                    if res["type"] in ["NuScenesDataset"]:
                        sem_labels = np.concatenate([sampled_sem_labels, sem_labels], axis=0)
                        base_index_mask = np.concatenate([sampled_base_indices, base_index_mask], axis=0)

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes
            
            gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )
            gt_dict["gt_boxes"], points = prep.global_translate_(
                gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std
            )
        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes


        if self.shuffle_points:
            print("shuffle points is off")
            raise NotImplementedError

        res["lidar"]["combined"] = points
        res["lidar"]["points"] = points

        if res["type"] in ["NuScenesDataset"]:
            res["lidar"]["base_index_mask"] = base_index_mask

        if self.mode == "train":
            if res["type"] in ["NuScenesDataset"]:
                res["lidar"]["panoptic_annotations"] = sem_labels
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"], max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        double_flip = self.double_flip and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )            

        return res, info

def flatten(box):
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return flatten(gt_classes)

@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        try:
            self._max_seg_points = assigner_cfg.max_seg_points
            self.no_segmentation = False
        except:
            # print("Not using segmentation heads")
            self.no_segmentation = True

        self.cfg = assigner_cfg

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]
        type_of_task = [t.get('task_type', 'detection') for t in self.tasks]
        all_classes = np.hstack(class_names_by_task)
        example = {}

        if res["mode"] == "train":
            # Calculate output featuremap size
            if 'voxels' in res['lidar']:
                # Calculate output featuremap size
                grid_size = res["lidar"]["voxels"]["shape"] 
                pc_range = res["lidar"]["voxels"]["range"]
                voxel_size = res["lidar"]["voxels"]["size"]
                feature_map_size = grid_size[:2] // self.out_size_factor
            else:
                pc_range = np.array(self.cfg['pc_range'], dtype=np.float32)
                voxel_size = np.array(self.cfg['voxel_size'], dtype=np.float32)
                grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
                grid_size = np.round(grid_size).astype(np.int64)

            feature_map_size = grid_size[:2] // self.out_size_factor

            gt_dict = res["lidar"]["annotations"]

            if res['type'] == 'NuScenesDataset':
                pano_gt_dict = {"sem_label": res["lidar"]["panoptic_annotations"]}
                run = (res["lidar"]["points"][res["lidar"]["base_index_mask"], -1]==0).sum() == pano_gt_dict["sem_label"].shape[0]
                assert(run)
            elif res['type'] == 'SemanticKITTIDataset':
                pass
            else:
                raise NotImplementedError


            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name_itr, class_name in enumerate(class_names_by_task):
                to_compare_dict = gt_dict["gt_classes"] if type_of_task[class_name_itr] == 'detection' else pano_gt_dict["sem_label"]
                task_masks.append(
                    [
                        np.where(
                            to_compare_dict == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            task_segments = []
            task_segments_names = []
            task_segments_classes = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                task_seg = []
                if type_of_task[idx] == 'detection':
                    for m in mask:
                        task_box.append(gt_dict["gt_boxes"][m])
                        task_class.append(gt_dict["gt_classes"][m] - flag2)
                        task_name.append(gt_dict["gt_names"][m])
                    task_boxes.append(np.concatenate(task_box, axis=0))
                    task_classes.append(np.concatenate(task_class))
                    task_names.append(np.concatenate(task_name))
                    flag2 += len(mask)
                elif type_of_task[idx] == 'segmentation':
                    flag3 = flag2
                    for m in mask[:-1]: #since last is other_background
                        task_seg.append(m[0])
                        task_class.append(np.ones(m[0].shape[0], dtype=np.int32)*(flag3 - flag2))
                        task_name.append([all_classes[flag3-1]]*m[0].shape[0])
                        flag3 += 1
                    flag2 += len(mask)
                    task_segments.append(np.concatenate(task_seg))
                    task_segments_classes.append(np.concatenate(task_class))
                    task_segments_names.append(np.concatenate(task_name))

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes
            
            if res['type'] == 'NuScenesDataset':
                pano_gt_dict["gt_classes"] = task_segments_classes
                pano_gt_dict["gt_names"] = task_segments_names
                pano_gt_dict["gt_segments"] = task_segments

            res["lidar"]["annotations"] = gt_dict

            if res['type'] == 'NuScenesDataset':
                res["lidar"]["panoptic_annotations"] = pano_gt_dict #could be removed if memory issue?

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                                dtype=np.float32)
                if type_of_task[idx] == 'detection':

                    if res['type'] == 'NuScenesDataset':
                        # [reg, hei, dim, vx, vy]
                        anno_box = np.zeros((max_objs, 8), dtype=np.float32)
                    elif res['type'] == 'WaymoDataset':
                        anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                    elif res['type'] == 'SemanticKITTIDataset':
                        anno_box = np.zeros((max_objs, 8), dtype=np.float32) 
                    else:
                        raise NotImplementedError("Only Support nuScene for Now!")

                    ind = np.zeros((max_objs), dtype=np.int64)
                    mask = np.zeros((max_objs), dtype=np.uint8)
                    cat = np.zeros((max_objs), dtype=np.int64)

                    num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)  

                    for k in range(num_objs):
                        cls_id = gt_dict['gt_classes'][idx][k] - 1

                        w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                                gt_dict['gt_boxes'][idx][k][5]
                        w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                        if w > 0 and l > 0:
                            radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                            radius = max(self._min_radius, int(radius))

                            # be really careful for the coordinate system of your box annotation. 
                            x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                    gt_dict['gt_boxes'][idx][k][2]

                            coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                            (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                            ct = np.array(
                                [coor_x, coor_y], dtype=np.float32)  
                            ct_int = ct.astype(np.int32)

                            # throw out not in range objects to avoid out of array area when creating the heatmap
                            if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                                continue 

                            draw_gaussian(hm[cls_id], ct, radius)

                            new_idx = k
                            x, y = ct_int[0], ct_int[1]

                            cat[new_idx] = cls_id
                            ind[new_idx] = y * feature_map_size[0] + x
                            mask[new_idx] = 1

                            if res['type'] == 'NuScenesDataset': 
                                # Now boxes are [center, size, vel]
                                vx, vy = gt_dict['gt_boxes'][idx][k][-2:]
                                # rot = gt_dict['gt_boxes'][idx][k][8]
                                anno_box[new_idx] = np.concatenate(
                                    (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                    np.array(vx), np.array(vy)), axis=None)
                            elif res['type'] == 'WaymoDataset':
                                vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                                rot = gt_dict['gt_boxes'][idx][k][-1]
                                anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                            elif res['type'] == 'SemanticKITTIDataset':
                                vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                                # rot = gt_dict['gt_boxes'][idx][k][-1]
                                anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy)), axis=None)
                            else:
                                raise NotImplementedError("Only Support Waymo and nuScene for Now")
                    
                    hms.append(hm)
                    anno_boxs.append(anno_box)
                    masks.append(mask)
                    inds.append(ind)
                    cats.append(cat)

                elif type_of_task[idx] == 'segmentation':
                    if res['type'] == 'NuScenesDataset':
                        ind = np.zeros((self._max_seg_points), dtype=np.int64)
                        mask = np.zeros((self._max_seg_points), dtype=np.uint8)
                        cat = np.zeros((self._max_seg_points), dtype=np.int64)

                        cls_id = task_segments_classes[idx - 6]
                        cls_points = res["lidar"]["points"][res["lidar"]["base_index_mask"]][task_segments[idx-6]][:, [0, 1]]
                        x, y = cls_points[:, 0], cls_points[:, 1]
                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                            (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        # Currently in order to avoid changing feature size, restricting labels to be within 50m.. same as detection.
                        valid_mask_x = (feature_map_size[0] > coor_x) & (coor_x >= 0) 
                        valid_mask_y = (feature_map_size[1] > coor_y) & (coor_y >= 0)
                        in_range_mask = valid_mask_x & valid_mask_y #retains about 98.5% of data #should change not points are clipped
                        cls_id, coor_y, coor_x = cls_id[in_range_mask], coor_y[in_range_mask], coor_x[in_range_mask]

                        ## set only unique. we don't want repeats.
                        ct = np.array(
                                [coor_x, coor_y], dtype=np.float32)  
                        ct_int = ct.astype(np.int32)
                        uniq_ct_int, uniq_index_ct_int = np.unique(ct_int, axis=1, return_index=True)
                        num_points = uniq_ct_int.shape[1]
                        hm[cls_id[uniq_index_ct_int], uniq_ct_int[1], uniq_ct_int[0]] = 1 #setting 1 ignores density i.e multiple points having same x,y. but is fast!

                        hm[-1] = (hm.sum(axis=0)==0).astype(hm.dtype)
                        hm = hm.argmax(axis=0)

                        assert(num_points < self._max_seg_points)
                        cat[:num_points] = cls_id[uniq_index_ct_int]
                        ind[:num_points] = uniq_ct_int[1] * feature_map_size[0] + uniq_ct_int[0]
                        mask[:num_points] = 1
                    else:
                        raise NotImplementedError

                    hms.append(hm)
                    masks.append(mask)
                    inds.append(ind)
                    cats.append(cat)

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})

            # used for two stage code 
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] == "WaymoDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 9), dtype=np.float32)
            elif res['type'] == "SemanticKITTIDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes, 
                classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            # used for per-point segmentation
            pt_sem_labs = -1 * np.ones(self._max_seg_points, dtype=np.float64)
            
            if res['lidar']['base_index_mask'].sum() > self._max_seg_points:
                assert res['lidar']['base_index_mask'].sum() <= self._max_seg_points
            
            pt_sem_labs[:res['lidar']['base_index_mask'].sum()] = pano_gt_dict['sem_label']

            # adding base_mask could be memory intensive, so skipping?
            base_index_mask = -1 * np.ones(8*self._max_seg_points, dtype=np.int8)
            base_index_mask[:len(res['lidar']['base_index_mask'])] = res['lidar']['base_index_mask'] * 1
            example.update({'pt_sem_labs': pt_sem_labs, 'base_index_mask': base_index_mask})

            # used for voxel segmentation, currently not tightly integrated with voxelization but should be
            # but based upon understanding of voxelization should work
            points = res["lidar"]["points"][res["lidar"]["base_index_mask"]]

            points[:, :3] = np.clip(points[:, :3], pc_range[:3], pc_range[3:])

            intervals = (pc_range[3:] - pc_range[:3]) / (grid_size//2 - 1)
            label_grid_ind = np.floor((points[:, [0,1,2]] - pc_range[[0,1,2]]) / (intervals)).astype(np.int)

            labels = pano_gt_dict['sem_label'].astype(np.int)
            label_voxel_pair = np.concatenate([label_grid_ind, labels[:, None]], axis=1)
            # the order in below sort doesn't matter as long as its sorted
            label_voxel_pair = label_voxel_pair[np.lexsort((label_grid_ind[:, 2], label_grid_ind[:, 1], label_grid_ind[:, 0])), :]
            # processed_label = np.zeros(grid_size, dtype=np.uint8)
            processed_label = np.zeros(grid_size//2, dtype=np.uint8)
            processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

            #can think of improving the above BEV labels with the voxel based.
            example.update({'voxel_sem_labs': processed_label})#, 'voxel_label_grid': label_grid_ind})

        else:
            pass

        res["lidar"]["targets"] = example

        # now the points can be shuffled if that helps and done correctly..
        return res, info

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label
