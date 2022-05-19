# Training of Monodepth2 on the nuScenes Dataset
## Introduction
This repository is to integrate the nuScenes and the Cityscapes datasets to the monodepth2 training.

## NuScenes
### Setup
Please check [nuscenes_depth](https://github.com/bolianchen/nuscenes_depth) about:
- [Download of the nuScenes dataset](https://github.com/bolianchen/nuscenes_depth#download-the-nuscenes-dataset)
- [The required Python modules](https://github.com/bolianchen/nuscenes_depth#environment-setup) in addition to [the ones required by monodepth2](https://github.com/nianticlabs/monodepth2#%EF%B8%8F-setup)
### Training Script & Results
<pre>
python train.py --data_path $NUSCENES_TRAINVAL
                --model_name nuscenes_mono2 --height 288 --width 512 
                --nuscenes_version v1.0-trainval
                --camera_channels CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT 
                                  CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT
                --speed_bound 2 inf
                
</pre>
<p align="center">
  <img src="assets/nuscenes_scene-0655.gif" width="600" />
</p>

## Cityscapes
[under development]
