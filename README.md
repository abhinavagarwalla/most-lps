# Lidar Panoptic Segmentation and Tracking without Bells and Whistles

This repository provides implementation for MOST, which generates panoptic predictions for input point clouds, as described in the paper [Lidar Panoptic Segmentation and Tracking without Bells and Whistles](https://mostlps.github.io/assets/paper.pdf).

![](https://mostlps.github.io/figures/teaser.png)

State-of-the-art lidar panoptic segmentation (LPS) methods follow bottom-up segmentation-centric fashion wherein they build upon semantic segmentation networks by utilizing clustering to obtain object instances. In this paper, we re-think this approach and propose a surprisingly simple yet effective detection-centric network for both LPS and tracking. Our network is modular by design and optimized for all aspects of both the panoptic segmentation and tracking task. One of the core components of our network is the object instance detection branch, which we train using point-level (modal) annotations, as available in segmentation-centric datasets. In the absence of amodal (cuboid) annotations, we regress modal centroids and object extent using trajectory-level supervision that provides information about object size, which cannot be inferred from single scans due to occlusions and the sparse nature of the lidar data. We obtain fine-grained instance segments by learning to associate lidar points with detected centroids. We evaluate our method on several 3D/4D LPS benchmarks and observe that our model establishes a new state-of-the-art among open-sourced models, outperforming recent query-based models.

## Dependencies and Installation

We refer to the CenterPoint's docs for installation.
Please refer to [Installation](https://github.com/tianweiy/CenterPoint/blob/master/docs/INSTALL.md) for setting up the required dependencies and the environment. 
Please refer to [NuScenes](https://github.com/tianweiy/CenterPoint/blob/master/docs/NUSC.md) documents for setting up NuScenes dataset.

## Usage

Generating modal annoations:
- `python tools/create_modal_boxes.py`

Starting the training on NuScenes:
- `python -m torch.distributed.launch --nproc_per_node=8 --master_port 98992 ./tools/train.py configs/nusc/encoder_decoder/nusc_centerpoint_encoderdecoder_0075voxel_fix_bn_z_scale_fade_instance_max.py --work_dir ./WORK_DIR`

Once the training finishes, we would obtain a model with that does BEV-level 3D model detection as well as voxel-level segmentation. In order to obtain point-wise predictions, we utilize PointSegMLP. We would be releasing the code for PointSegMLP shortly.

## Results
![](assets/supp_video.gif)

## License

See the [LICENSE](https://github.com/abhinavagarwalla/most-lps/blob/master/LICENSE) file for more details.

## Citation

If you find this work useful in your research, please consider citing the paper:

```
@article{MOST23,
    title = {Lidar Panoptic Segmentation and Tracking without Bells and Whistles},
    author = {Agarwalla, Abhinav and Huang, Xuhua and Ziglar, Jason and Ferroni,Francesco and Leal-Taixé.Laura and Hays, James and Ošep, Aljoša and Ramanan,Deva},
    journal = {IROS},
    Year = {2023}
}