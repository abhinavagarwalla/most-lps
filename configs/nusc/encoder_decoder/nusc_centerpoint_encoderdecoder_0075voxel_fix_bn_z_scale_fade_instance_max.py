import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=1, task_type='detection', class_names=["car"]),
    dict(num_class=2, task_type='detection', class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, task_type='detection', class_names=["bus", "trailer"]),
    dict(num_class=1, task_type='detection', class_names=["barrier"]),
    dict(num_class=2, task_type='detection', class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, task_type='detection', class_names=["pedestrian", "traffic_cone"]),
    dict(num_class=7, task_type='segmentation', class_names=["driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation", "other_background"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="EncoderDecoderNetV2",
    pretrained=None,
    reader=dict(
        type="DynamicVoxelPPEncoderClip",
        pc_range=[-54, -54, -5.0, 54, 54, 3.0],
        voxel_size=[0.075, 0.075, 0.2],
        num_input_features=5,
        num_output_filters=32,
    ),
    backbone=dict(
        type="Asymm_3d_spconv", num_input_features=5+32, ds_factor=8
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=False
    ),
    refiner=dict(
        type='VoxelPredRefiner', in_channels=[512+32+5, 64], 
        num_classes=17
    )
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    max_seg_points = 1000000,
    pc_range=[-54, -54, -5.0, 54, 54, 3.0],
    voxel_size=[0.075, 0.075, 0.2]
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-54, -54],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.075, 0.075]
)

# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10
data_root = "/data/datasets/nuscenes"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path=data_root + "/dbinfos_train_10sweeps_withvelo_modalvel_instance_max.pkl",
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=2),
        dict(motorcycle=6),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=False,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.9, 1.1],
    global_translate_std=0.5,
    db_sampler=db_sampler,
    class_names=class_names,
    no_augmentation=False
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True, dataroot=data_root, version='v1.0-trainval'),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True, dataroot=data_root, version='v1.0-trainval'),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = data_root + "/infos_train_10sweeps_withvelo_filter_True_modalvel_instance_max.pkl"
val_anno = data_root + "/infos_val_10sweeps_withvelo_filter_True_modal.pkl"
test_anno = data_root + "/infos_test_10sweeps_withvelo.pkl"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        # version='v1.0-mini'
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        test_mode=True,
        class_names=class_names,
        pipeline=test_pipeline,
        version='v1.0-test'
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]

learning_map = {1: 0,5: 0,7: 0,8: 0,10: 0,11: 0,13: 0,19: 0,20: 0,0: 0,29: 0,31: 0,9: 1,14: 2,15: 3,16: 3,17: 4,18: 5,21: 6,2: 7,3: 7,4: 7,6: 7,12: 8,22: 9,23: 10,24: 11,25: 12,26: 13,27: 14,28: 15,30: 16}
