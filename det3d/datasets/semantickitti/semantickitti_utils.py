import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from os import makedirs
import inspect



import os.path
import shutil
import sys
import tempfile
import yaml
from collections import abc
from importlib import import_module
from addict import Dict


def make_dir(folder_name):
    """Create a directory.

    If already exists, do nothing
    """
    if not exists(folder_name):
        makedirs(folder_name)


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, abc.Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
        else:
            print(f'cannot parse key {prefix + k} of type {type(v)}')
    return parser


class Config(object):

    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict should be a dict, but'
                            f'got {type(cfg_dict)}')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))

        self.cfg_dict = cfg_dict

    def dump(self, *args, **kwargs):
        """Dump to a string."""

        def convert_to_dict(cfg_node, key_list):
            if not isinstance(cfg_node, ConfigDict):
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict

        self_as_dict = convert_to_dict(self._cfg_dict, [])
        print(self_as_dict)
        return yaml.dump(self_as_dict, *args, **kwargs)
        #return self_as_dict

    @staticmethod
    def merge_cfg_file(cfg, args, extra_dict):
        """Merge args and extra_dict from the input arguments.

        Merge the dict parsed by MultipleKVAction into this cfg.
        """
        # merge args to cfg
        if args.device is not None:
            cfg.pipeline.device = args.device
            cfg.model.device = args.device
        if args.split is not None:
            cfg.pipeline.split = args.split
        if args.main_log_dir is not None:
            cfg.pipeline.main_log_dir = args.main_log_dir
        if args.dataset_path is not None:
            cfg.dataset.dataset_path = args.dataset_path
        if args.ckpt_path is not None:
            cfg.model.ckpt_path = args.ckpt_path

        extra_cfg_dict = {'model': {}, 'dataset': {}, 'pipeline': {}}

        for full_key, v in extra_dict.items():
            d = extra_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict_dataset = Config._merge_a_into_b(extra_cfg_dict['dataset'],
                                                  cfg.dataset)
        cfg_dict_pipeline = Config._merge_a_into_b(extra_cfg_dict['pipeline'],
                                                   cfg.pipeline)
        cfg_dict_model = Config._merge_a_into_b(extra_cfg_dict['model'],
                                                cfg.model)

        return cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model

    @staticmethod
    def merge_module_cfg_file(args, extra_dict):
        """Merge args and extra_dict from the input arguments.

        Merge the dict parsed by MultipleKVAction into this cfg.
        """
        # merge args to cfg
        cfg_dataset = Config.load_from_file(args.cfg_dataset)
        cfg_model = Config.load_from_file(args.cfg_model)
        cfg_pipeline = Config.load_from_file(args.cfg_pipeline)

        cfg_dict = {
            'dataset': cfg_dataset.cfg_dict,
            'model': cfg_model.cfg_dict,
            'pipeline': cfg_pipeline.cfg_dict
        }
        cfg = Config(cfg_dict)

        if args.device is not None:
            cfg.pipeline.device = args.device
        if args.split is not None:
            cfg.pipeline.split = args.split
        if args.main_log_dir is not None:
            cfg.pipeline.main_log_dir = args.main_log_dir
        if args.dataset_path is not None:
            cfg.dataset.dataset_path = args.dataset_path

        extra_cfg_dict = {'model': {}, 'dataset': {}, 'pipeline': {}}

        for full_key, v in extra_dict.items():
            d = extra_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict_dataset = Config._merge_a_into_b(cfg.dataset,
                                                  extra_cfg_dict['dataset'])
        cfg_dict_pipeline = Config._merge_a_into_b(cfg.pipeline,
                                                   extra_cfg_dict['pipeline'])
        cfg_dict_model = Config._merge_a_into_b(cfg.model,
                                                extra_cfg_dict['model'])

        return cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model

    @staticmethod
    def _merge_a_into_b(a, b):
        # merge dict `a` into dict `b` (non-inplace). values in `a` will
        # overwrite `b`.
        # copy first to avoid inplace modification
        # from mmcv mmcv/utils/config.py
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                if not isinstance(b[k], dict):
                    raise TypeError(
                        "{}={} in child config cannot inherit from base ".
                        format(k, v) +
                        "because {} is a dict in the child config but is of ".
                        format(k) +
                        "type {} in base config.  ".format(type(b[k])))
                b[k] = Config._merge_a_into_b(v, b[k])
            else:
                if v is None:
                    continue
                if v.isnumeric():
                    v = int(v)
                elif v.replace('.', '').isnumeric():
                    v = float(v)
                elif v == 'True' or v == 'true':
                    v = True
                elif v == 'False' or v == 'false':
                    v = False
                b[k] = v
        return b

    def merge_from_dict(self, new_dict):
        """Merge a new into cfg_dict.

        Args:
            new_dict (dict): a dict of configs.
        """
        b = self.copy()
        for k, v in new_dict.items():
            if v is None:
                continue
            b[k] = v
        return Config(b)

    @staticmethod
    def load_from_file(filename):
        if filename is None:
            return Config()
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} not found')

        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix='.py')
                temp_config_name = os.path.basename(temp_config_file.name)
                shutil.copyfile(filename,
                                os.path.join(temp_config_dir, temp_config_name))
                temp_module_name = os.path.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]
                # close temp file
                temp_config_file.close()

        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filename) as f:
                cfg_dict = yaml.safe_load(f)

        return Config(cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)


class BaseDataset(ABC):
    """The base dataset class that is used by all other datasets.

    All datasets must inherit from this class and implement the functions in order to be
    compatible with pipelines.

    Args:
        **kwargs: The configuration of the model as keyword arguments.

    Attributes:
        cfg: The configuration file as Config object that stores the keyword
            arguments that were passed to the constructor.
        name: The name of the dataset.

    **Example:**
        This example shows a custom dataset that inherit from the base_dataset class:

            from .base_dataset import BaseDataset

            class MyDataset(BaseDataset):
            def __init__(self,
                 dataset_path,
                 name='CustomDataset',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[],
                 test_result_folder='./test',
                 val_files=['Custom.ply'],
                 **kwargs):
    """

    def __init__(self, **kwargs):
        """Initialize the class by passing the dataset path."""
        if kwargs['dataset_path'] is None:
            raise KeyError("Provide dataset_path to initialize the dataset")

        if kwargs['name'] is None:
            raise KeyError("Provide dataset name to initialize it")

        self.cfg = Config(kwargs)
        self.name = self.cfg.name

    @staticmethod
    @abstractmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """

    @abstractmethod
    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return

    @abstractmethod
    def is_tested(self, attr):
        """Checks whether a datum has been tested.

        Args:
            attr: The attributes associated with the datum.

        Returns:
            This returns True if the test result has been stored for the datum with the
            specified attribute; else returns False.
        """
        return False

    @abstractmethod
    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        return

class BaseDatasetSplit(ABC):
    """The base class for dataset splits.

    This class provides access to the data of a specified subset or split of a dataset.

    Args:
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split, usually one of
            'training', 'test', 'validation', or 'all'.

    Attributes:
        cfg: Shortcut to the config of the dataset object.
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split, usually one of
            'training', 'test', 'validation', or 'all'.
    """

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        self.path_list = path_list
        self.split = split
        self.dataset = dataset

        from det3d.datasets.samplers import SemSegSpatiallyRegularSampler, SemSegRandomSampler
        if split in ['test']:
            sampler_cls = SemSegSpatiallyRegularSampler
        else:
            sampler_cls = SemSegRandomSampler
        self.sampler = sampler_cls(self)

    @abstractmethod
    def __len__(self):
        """Returns the number of samples in the split."""
        return 0

    @abstractmethod
    def get_data(self, idx):
        """Returns the data for the given index."""
        return {}

    @abstractmethod
    def get_attr(self, idx):
        """Returns the attributes for the given index."""
        return {}
