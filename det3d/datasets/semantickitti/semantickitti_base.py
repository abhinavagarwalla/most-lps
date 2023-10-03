import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
import yaml
from pathlib import Path

from .semantickitti_utils import BaseDataset, BaseDatasetSplit, make_dir
from .bev_box import BEVBoundingBox3D

class SemanticKITTI(BaseDataset):
    """This class is used to create a dataset based on the SemanticKitti
    dataset, and used in visualizer, training, or testing.

    The dataset is best for semantic scene understanding.
    """

    def __init__(self,
                 dataset_path,
                 name='SemanticKITTI',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 class_weights=[
                     55437630, 320797, 541736, 2578735, 3274484, 552662, 184064,
                     78858, 240942562, 17294618, 170599734, 6369672, 230413074,
                     101130274, 476491114, 9833174, 129609852, 4506626, 1168181
                 ],
                 ignored_label_inds=[0],
                 test_result_folder='./test',
                 test_split=[
                     '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                     '21'
                 ],
                 training_split=[
                     '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
                 ],
                 validation_split=['08'],
                 all_split=[
                     '00', '01', '02', '03', '04', '05', '06', '07', '09', '08',
                     '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                     '20', '21'
                 ],
                 return_reflectance=False,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         test_split=test_split,
                         training_split=training_split,
                         validation_split=validation_split,
                         all_split=all_split,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)

        data_config = join(dirname(abspath(__file__)), '_resources/',
                           'semantic-kitti.yaml')
        DATA = yaml.safe_load(open(data_config, 'r'))
        remap_dict = DATA["learning_map_inv"]

        # make lookup table for mapping
        max_key = max(remap_dict.keys())
        remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

        remap_dict_val = DATA["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(
            remap_dict_val.values())

        self.remap_lut_val = remap_lut_val
        self.remap_lut = remap_lut

        self.return_reflectance = return_reflectance

        max_sem_key = 0
        sem_color_dict = DATA['color_map']
        for key, data in sem_color_dict.items():
            if key + 1 > max_sem_key:
                max_sem_key = key + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for key, value in sem_color_dict.items():
            self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'unlabeled',
            1: 'car',
            2: 'bicycle',
            3: 'motorcycle',
            4: 'truck',
            5: 'other-vehicle',
            6: 'person',
            7: 'bicyclist',
            8: 'motorcyclist',
            9: 'road',
            10: 'parking',
            11: 'sidewalk',
            12: 'other-ground',
            13: 'building',
            14: 'fence',
            15: 'vegetation',
            16: 'trunk',
            17: 'terrain',
            18: 'pole',
            19: 'traffic-sign'
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return SemanticKITTISplit(self, split=split)

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        name_seq, name_points = name.split("_")
        test_path = join(cfg.test_result_folder, 'sequences')
        save_path = join(test_path, name_seq, 'predictions')
        test_file_name = name_points
        store_path = join(save_path, name_points + '.label')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name']
        name_seq, name_points = name.split("_")

        test_path = join(cfg.test_result_folder, 'sequences')
        make_dir(test_path)
        save_path = join(test_path, name_seq, 'predictions')
        make_dir(save_path)
        test_file_name = name_points
        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(save_path, name_points + '.label')

        pred = self.remap_lut[pred].astype(np.uint32)
        pred.tofile(store_path)

    def save_test_result_kpconv(self, results, inputs):
        cfg = self.cfg
        for j in range(1):
            name = inputs['attr']['name']
            name_seq, name_points = name.split("_")

            test_path = join(cfg.test_result_folder, 'sequences')
            make_dir(test_path)
            save_path = join(test_path, name_seq, 'predictions')
            make_dir(save_path)

            test_file_name = name_points

            proj_inds = inputs['data'].reproj_inds[0]
            probs = results[proj_inds, :]

            pred = np.argmax(probs, 1)

            store_path = join(save_path, name_points + '.label')
            pred = pred + 1
            pred = self.remap_lut[pred].astype(np.uint32)
            pred.tofile(store_path)

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split in ['train', 'training']:
            seq_list = cfg.training_split
        elif split in ['test', 'testing']:
            seq_list = cfg.test_split
        elif split in ['val', 'validation']:
            seq_list = cfg.validation_split
        elif split in ['all']:
            seq_list = cfg.all_split
        else:
            raise ValueError("Invalid split {}".format(split))

        for seq_id in seq_list:
            pc_path = join(dataset_path, 'dataset', 'sequences', seq_id,
                           'velodyne')
            file_list.append(
                [join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        file_list = np.concatenate(file_list, axis=0)

        return file_list

    @staticmethod
    def _extend_matrix(mat):
        mat = np.concatenate(
        [mat, np.array([[0., 0., 1., 0.]], dtype=mat.dtype)], axis=0)
        return mat

    @staticmethod
    def read_timestamps(path):
        assert Path(path).exists()
        times = list(map(lambda x: float(x.strip()), open(path, 'r').readlines()))
        return times

    @staticmethod
    def read_calib(path):
        """Reads calibiration for the dataset. You can use them to compare
        modeled results to observed results.

        Returns:
        The camera and the camera image used in calibration.
        """
        assert Path(path).exists()

        with open(path, 'r') as f:
            lines = f.readlines()

        obj = lines[0].strip().split(' ')[1:]
        P0 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[1].strip().split(' ')[1:]
        P1 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32).reshape(3, 4)

        P0 = SemanticKITTI._extend_matrix(P0)
        P1 = SemanticKITTI._extend_matrix(P1)
        P2 = SemanticKITTI._extend_matrix(P2)
        P3 = SemanticKITTI._extend_matrix(P3)

        obj = lines[4].strip().split(' ')[1:]
        Tr = np.eye(4, dtype=np.float32)
        Tr[:3] = np.array(obj, dtype=np.float32).reshape(3, 4)
        
        world_cam = np.transpose(Tr)
        cam_img = np.transpose(P2)

        return {'world_cam': world_cam, 'cam_img': cam_img}

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut=None):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF # semantic label in lower half
        inst_label = label# >> 16 # instance id in upper half, no shifting to ensure uniqueness
        assert ((sem_label + ((inst_label>>16) << 16) == label).all())
        if remap_lut is not None:
            sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32), inst_label.astype(np.int32)

    @staticmethod
    def load_index_kitti(index_path):
        index = np.fromfile(index_path, dtype=np.int8).reshape((-1, 1))
        return index

    @staticmethod
    def load_flow_kitti(flow_path):
        flow = np.fromfile(flow_path, dtype=np.float32).reshape((-1, 3))
        return flow

    @staticmethod
    def get_bounding_boxes(points, labels, inst_labels, remap_lut_val,
        min_points=10, method='hybrid-select'):#

        # Adjust min_points according to pcd density; raw vs dense
        VALID_THINGS_LABELS = remap_lut_val[[10, 11, 13, 15, 18, 20, 30, 31, 32]]
        things_mask = [l_i for l_i, l_v in enumerate(labels) if l_v in VALID_THINGS_LABELS]

        points = points[things_mask]
        labels = labels[things_mask]
        inst_labels = inst_labels[things_mask]

        uniq_inst, uniq_rev, uniq_count = np.unique(inst_labels, return_counts=True, return_inverse=True)

        boxes = []
        for u_i, u_v in enumerate(uniq_inst):
            if uniq_count[u_i] < min_points:
                continue
    
            #get all points for this instance
            in_points = points[uniq_rev == u_i]

            if method == 'zero_out_z':
                in_points_z = in_points.copy()
                in_points_z[:, 2] = np.random.randn(in_points_z.shape[0])
                pc = o3d.utility.Vector3dVector(in_points_z[:, :3])
                box = o3d.geometry.OrientedBoundingBox.create_from_points(pc)
                box3d = BEVBox3D(box.get_center(), box.extent,
                        DataProcessing.get_yaw(box.R, 2, 1, 0)[0],
                        u_v & 0xFFFF, 1.0)
            elif method == '9dof':
                pc = o3d.utility.Vector3dVector(in_points[:, :3])
                box = o3d.geometry.OrientedBoundingBox.create_from_points(pc)
                box3d = BEVBox3D(box.get_center(), box.extent,
                        DataProcessing.get_yaw(box.R, 2, 1, 0)[0],
                        u_v & 0xFFFF, 1.0)
            else:
                pc = in_points[:, :3]
                box3d = BEVBoundingBox3D.create_from_points(pc, method=method)
                if box3d is None:
                    continue
                
                box3d.label_class = remap_lut_val[u_v & 0xFFFF]

            boxes.append(box3d)

        # break
        return boxes

    @staticmethod
    def get_modal_velocity(idx, inst_label, cur_center, dir, file, max_time_diff=1.5):
        #try to get prev
        has_prev = int(file[:-4]) != 0
        has_next = str(int(file[:-4]) + 1).zfill(6) + '.bin' in os.listdir(dir)


        if has_prev:
            prev_idx = str(int(file[:-4]) - 1).zfill(6)
            prev_pcd = SemanticKITTI.load_pc_kitti(join(dir, prev_idx + '.bin'))
            _, prev_inst_labels = SemanticKITTI.load_label_kitti(
                join(dir, '../labels/', prev_idx + '.label'))
            if inst_label in prev_inst_labels:
                prev_center = np.mean(prev_pcd[prev_inst_labels == inst_label, :3], axis=0)
            else:
                has_prev = False

        if has_next:
            next_idx = str(int(file[:-4]) + 1).zfill(6)
            next_pcd = SemanticKITTI.load_pc_kitti(join(dir, next_idx + '.bin'))
            _, next_inst_labels = SemanticKITTI.load_label_kitti(
                join(dir, '../labels', next_idx + '.label'))
            if inst_label in next_inst_labels:
                next_center = np.mean(next_pcd[next_inst_labels == inst_label, :3], axis=0)
            else:
                has_next = False

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if not has_prev:
            prev_idx = file[:-4]
            prev_center = cur_center
        if not has_next:
            next_idx = file[:-4]
            next_center = cur_center
        
        times = SemanticKITTI.read_timestamps(join(dir, '../times.txt'))
        prev_time = times[int(prev_idx)]
        next_time = times[int(next_idx)]
        pos_diff = next_center - prev_center
        time_diff = next_time - prev_time

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2
            
        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    @staticmethod
    def get_center_radius_vel(idx, points, labels, inst_labels, remap_lut_val, dir, file,
        min_points=10, method='hybrid-select'):
        #Adjust min_points according to pcd density; raw vs dense

        # filter out stuff pixels
        VALID_THINGS_LABELS = remap_lut_val[[10, 11, 13, 15, 18, 20, 30, 31, 32]]
        things_mask = [l_i for l_i, l_v in enumerate(labels) if l_v in VALID_THINGS_LABELS]

        points = points[things_mask]
        labels = labels[things_mask]
        inst_labels = inst_labels[things_mask]

        uniq_inst, uniq_rev, uniq_count = np.unique(inst_labels, return_counts=True, return_inverse=True)

        boxes = []
        for u_i, u_v in enumerate(uniq_inst):
            if uniq_count[u_i] < min_points:
                continue
            
            #get all points for this instance
            in_points = points[uniq_rev == u_i]

            modal_center = np.mean(in_points, axis=0)
            modal_radius = np.sqrt(np.max((in_points - modal_center)**2, axis=0))
            modal_radius = np.clip(modal_radius, 0.05, None) # keep minimum to some value

            box3d = BEVBoundingBox3D(modal_center[:3], modal_radius[:3],
                        0., remap_lut_val[u_v & 0xFFFF])
            box3d.velocity = SemanticKITTI.get_modal_velocity(idx, u_v, modal_center[:3], dir, file)
            boxes.append(box3d)
        # visualize_prediction(points, labels, boxes, remap_lut_val)
        return boxes

class SemanticKITTISplit(BaseDatasetSplit):

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        print("Found {} pointclouds for {}".format(len(self.path_list),
                            split))
        self.remap_lut_val = dataset.remap_lut_val

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx, return_boxes=True):
        pc_path = self.path_list[idx]
        points = SemanticKITTI.load_pc_kitti(pc_path)

        dir, file = split(pc_path)
        label_path = join(dir, '../labels', file[:-4] + '.label')
        index_path = join(dir, '../indices', file[:-4] + '.index')
        flow_path = join(dir, '../flows', file[:-4] + '.flow' )
        if not exists(label_path):
            labels = np.zeros(np.shape(points)[0], dtype=np.int32)
        
        if self.split not in ['test', 'all'] and not os.path.exists(label_path):
            raise FileNotFoundError(f' Label file {label_path} not found')

        else:
            labels, inst_labels = SemanticKITTI.load_label_kitti(
                label_path, self.remap_lut_val)
            
            if exists(flow_path):
                raise NotImplementedError
            else:
                if return_boxes:
                    bounding_boxes = SemanticKITTI.get_center_radius_vel(
                        idx, points, labels, inst_labels, self.remap_lut_val, dir, file
                    )
                else:
                    bounding_boxes = None

    
        calib_path = join(dir, '../calib.txt')
        calib = SemanticKITTI.read_calib(calib_path)

        if self.dataset.return_reflectance:
            if not exists(index_path):
                data = {
                'point': points,
                'feat': None,
                'label': labels,
                'inst_label': inst_labels,
                'calib': calib,
                'bounding_boxes': bounding_boxes
                }
            else:
                indices = SemanticKITTI.load_index_kitti(index_path)

                data = {
                'point': points,
                'feat': None,
                'label': labels,
                'inst_label': inst_labels,
                'calib': calib,
                'bounding_boxes': bounding_boxes,
                'indices': indices
                }
        else:
            data = {
                'point': points[:, 0:3],
                'feat': None,
                'label': labels,
                'inst_label': inst_labels,
                'calib': calib,
                'bounding_boxes': bounding_boxes,
            }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        dir, file = split(pc_path)
        _, seq = split(split(dir)[0])
        name = '{}_{}'.format(seq, file[:-4])

        pc_path = str(pc_path)
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': self.split}
        return attr

