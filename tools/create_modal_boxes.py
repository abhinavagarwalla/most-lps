import copy
import os
import pickle

import numpy as np
import open3d as o3d
from bleach import clean
from matplotlib import pyplot as plt
from ml3d.datasets import NuScenes
from tqdm import tqdm


def modal_mean_fill(infos, output_path=None):
    # Compute average/min for each class from amodal; assign that as the minimum.
    mean_cls_sizes = {}
    for m in infos:
        # just keep sum and count.
        for obj_i, obj in enumerate(m['gt_boxes']):
            cls_name = m['gt_names'][obj_i]
            if cls_name in mean_cls_sizes.keys():
                if np.max(obj[3:6]) > 0.06:
                    mean_cls_sizes[cls_name][:3] += obj[3:6]
                    mean_cls_sizes[cls_name][3] += 1
            else:
                mean_cls_sizes[cls_name] = np.zeros(4)

    # print out average sizes
    for cls, stats in mean_cls_sizes.items():
        print(cls, stats, stats[:3]/stats[3])
        mean_cls_sizes[cls] = stats[:3]/stats[3]

    if output_path is not None:
        # Save the imputed info files to the output path
        for m in infos:
            for obj_i, obj in enumerate(m['gt_boxes']):
                if np.max(obj[3:6]) > 0.06:
                    continue
                else:
                    cls_name = m['gt_names'][obj_i]
                    m['gt_boxes'][obj_i][3:6] = mean_cls_sizes[cls_name]
        pickle.dump(infos, open(output_path, 'wb'))


def instance_max_fill(infos, output_path=None):
    # Compute max for each instance; assign that as the extents.
    dataset = NuScenes(dataset_path='./datasets/nuscenes/',
                                    version='v1.0-trainval')

    # Compute the maximum size for that instance; and backfill
    max_inst_sizes = {}
    for m in infos:
        for obj_i, obj in enumerate(m['gt_boxes']):
            obj_token = m['gt_boxes_token'][obj_i]
            prev_obj_token = dataset.nusc.get(
                'sample_annotation', obj_token)['prev']

            # saves the max till now
            if prev_obj_token in max_inst_sizes.keys():
                max_inst_sizes[obj_token] = np.maximum(
                    obj[3:6], max_inst_sizes[prev_obj_token])
            else:
                max_inst_sizes[obj_token] = obj[3:6]

    # backfill now
    for m in infos[::-1]:
        for obj_i, obj in enumerate(m['gt_boxes']):
            obj_token = m['gt_boxes_token'][obj_i]
            prev_obj_token = dataset.nusc.get(
                'sample_annotation', obj_token)['prev']
            if prev_obj_token != '':
                max_inst_sizes[prev_obj_token] = max_inst_sizes[obj_token]

            m['gt_boxes'][obj_i][3:6] = max_inst_sizes[obj_token]

    if output_path is not None:
        # Save the imputed info files to the output path
        pickle.dump(infos, open(output_path, 'wb'))


def get_points_inside_radius(points, radius, center):
    if type(radius) in [list, np.ndarray]:
        axis = np.abs(points[:, :3] - center)
        dots = (axis < np.abs(np.asarray(radius))).all(axis=1)
    else:
        axis = np.sqrt(np.sum(np.square(points[:, :3] - center), axis=1))
        dots = axis <= radius
    return dots


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def iterative_icp_fill(infos, output_path=None, threshold=0.1):
    # Read calibration files for initial transformation!
    # Read point cloud file for getting the actual points
    # Keep a set of target points for all objects.
    # Compute the extents for the objects when all the processing is done.
    dataset = NuScenes(dataset_path='./datasets/nuscenes/',
                                    version='v1.0-trainval')
    train_split = dataset.get_split("training")

    data = train_split.get_data(0)

    object_store = {}
    for d_i in tqdm(range(len(train_split))):
        # load point cloud
        data = train_split.get_data(d_i)
        m = infos[d_i]
        assert(data['token'] == m['token'])

        points = data['point']
        inst_label = data['inst_label']

        num_pred_boxes = infos[d_i]['gt_boxes'].shape[0]
        processed = np.zeros(num_pred_boxes, dtype=np.bool)

        infos[d_i]['gt_boxes'] = np.append(
            infos[d_i]['gt_boxes'], np.zeros((num_pred_boxes, 1)), axis=1)
        for u_i, id in enumerate(np.unique(inst_label)):
            # ignore stuff/ignore instances
            if id == 0 or dataset.remap_lut_val[id // 1000] == 0 or dataset.remap_lut_val[id // 1000] >= 11:
                continue

            points_ins = points[inst_label == id]
            point_ins_mean = np.mean(points_ins, axis=0)

            # get nearest center of same class
            candidate_boxes_dist = np.square(
                infos[d_i]['gt_boxes'][:, :3] - point_ins_mean[:3]).sum(axis=1)
            candidate_boxes_dist[processed == True] = np.inf
            candidate_boxes_dist[infos[d_i]['gt_names'] !=
                                 dataset.labels_mapping_16[dataset.remap_lut_val[id // 1000]]] = np.inf
            assert(np.min(candidate_boxes_dist) < np.inf)
            assert(np.min(candidate_boxes_dist) < 1.0)  # should be very close

            box_id = np.argmin(candidate_boxes_dist)
            box = infos[d_i]['gt_boxes'][box_id]
            processed[box_id] = True

            obj_token = m['gt_boxes_token'][box_id]
            prev_obj_token = dataset.nusc.get(
                'sample_annotation', obj_token)['prev']

            # store the transformations
            if prev_obj_token in object_store.keys():
                # do transformations on previous point cloud. by running ICP
                source_pcd = object_store[prev_obj_token]
                source_points = np.asarray(source_pcd['points'].points)

                points_homogenous = np.append(points_ins[:, :3], np.ones(
                    points_ins.shape[0])[:, None], axis=1)
                global_from_ref = np.linalg.inv(
                    np.dot(m['ref_from_car'], m['car_from_global']))
                target_points = (global_from_ref @
                                 points_homogenous.T).T[:, :3]
                target_points[:, 2] = 0.

                target_pcd = o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(target_points))
                target_center = np.mean(target_points, axis=0)

                # intialize transform as center translation
                trans_init = np.eye(4)
                trans_init[:3, 3] = target_center - source_pcd['center']

                if False or dataset.nusc.get('sample_annotation', obj_token)['next'] == '':
                    print("Source points: ", source_points, source_points.shape)
                    print("Target points: ", target_points, target_points.shape)
                    draw_registration_result(
                        source_pcd['points'], target_pcd, trans_init)

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source_pcd['points'], target_pcd, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(
                        with_scaling=False),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=2000)
                )

                if False or dataset.nusc.get('sample_annotation', obj_token)['next'] == '':
                    draw_registration_result(
                        source_pcd['points'], target_pcd, reg_p2p.transformation)

                global_pcd = o3d.geometry.PointCloud()
                transformed_source_points = np.append(
                    source_points, np.ones(source_points.shape[0])[:, None], axis=1)
                transformed_source_points = (
                    reg_p2p.transformation @ transformed_source_points.T).T[:, :3]
                global_pcd.points = o3d.utility.Vector3dVector(
                    np.append(target_points, transformed_source_points, axis=0))

                object_store[obj_token] = {'points': global_pcd,
                                           'center': np.mean(global_pcd.points, axis=0)
                                           }
            else:
                points_homogenous = np.append(points_ins[:, :3], np.ones(
                    points_ins.shape[0])[:, None], axis=1)
                global_from_ref = np.linalg.inv(
                    np.dot(m['ref_from_car'], m['car_from_global']))
                global_points = (global_from_ref @
                                 points_homogenous.T).T[:, :3]
                # project to 2D
                global_points[:, 2] = 0.
                global_pcd = o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(global_points))
                object_store[obj_token] = {'points': global_pcd,
                                           'center': np.mean(global_points, axis=0)
                                           }


def get_num_points_inside_radius(points, radius, center):
    if type(radius) in [list, np.ndarray]:
        axis = np.abs(points[:, :3] - center)
        dots = (axis < np.abs(np.asarray(radius))).all(axis=1)
    else:
        axis = np.sqrt(np.sum(np.square(points[:, :3] - center), axis=1))
        dots = axis <= radius
    return dots.sum()


def drop_min_points(infos, output_path=None):
    # read point cloud
    # compute number of inside points
    # remove instances with less than 3 points.

    dataset = NuScenes(dataset_path='./datasets/nuscenes/',
                                    version='v1.0-trainval')
    train_split = dataset.get_split("training")

    clean_modal_infos = []

    for d_i in tqdm(range(len(train_split))):
        # load point cloud
        data = train_split.get_data(d_i)

        points = data['point']

        assert(data['token'] == infos[d_i]['token'])
        clean_modal_info = {}
        for k, v in infos[d_i].items():
            if k not in ['gt_boxes_velocity', 'gt_names', 'gt_boxes_token', 'gt_boxes']:
                clean_modal_info[k] = v
            else:
                clean_modal_info[k] = []

        for obj_i, obj in enumerate(infos[d_i]['gt_boxes']):
            # compute num_inside_points
            num_inside_points = get_num_points_inside_radius(
                points, obj[3:6], obj[0:3])

            if num_inside_points < 3:
                continue
            else:
                for key in ['gt_boxes_velocity', 'gt_names', 'gt_boxes_token', 'gt_boxes']:
                    clean_modal_info[key].append(infos[d_i][key][obj_i])

        for key in ['gt_boxes_velocity', 'gt_names', 'gt_boxes_token', 'gt_boxes']:
            clean_modal_info[key] = np.array(clean_modal_info[key])
        clean_modal_infos.append(clean_modal_info)

    if output_path is not None:
        # Save the imputed info files to the output path
        pickle.dump(clean_modal_infos, open(output_path, 'wb'))


def fill_modal_infos(infos, strategy='modal_mean', output_path=None):
    if strategy == 'modal_mean':
        modal_mean_fill(infos, output_path)

    elif strategy == 'instance_max':
        instance_max_fill(infos, output_path)

    elif strategy == 'drop_min_points':
        drop_min_points(infos, output_path)

    elif strategy == 'iterative_icp':
        iterative_icp_fill(infos, output_path)
    else:
        raise Exception("Strategy not available.")


if __name__ == "__main__":
    ROOT_PATH = './datasets/nuscenes/'
    info_pickle = 'infos_train_10sweeps_withvelo_filter_True.pkl'
    infos = pickle.load(open(os.path.join(info_pickle), 'rb'))
    fill_modal_infos(infos, strategy='instance_max',
                    output_path=info_pickle.replace('.pkl', 'modalvel_instance_max.pkl'))
