# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth",
                                     "kitti_test", "nuscenes"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=512)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        ## DATA AUGMENTATION OPTIONS
        self.parser.add_argument("--not_do_color_aug",
                                 help="whether to do color augmentation",
                                 action="store_true")
        self.parser.add_argument("--not_do_flip",
                                 help="whether to flip image horizontally ",
                                 action="store_true")
        self.parser.add_argument("--do_crop",
                                 help="whether to crop image height",
                                 action="store_true")
        self.parser.add_argument("--crop_bound",
                                 type=float, nargs="+",
                                 help="for example, crop_bound=[0.0, 0.8]"
                                      " means the bottom 20% of the image will"
                                      " never be cropped. If only one value is"
                                      " given, only the top will be cropped"
                                      " according to the ratio",
                                 default=[0.0, 1.0])

        ## POSSIBLY MOBILE MASKS options
        MASK = ["none", "mono", "color"]
        self.parser.add_argument("--seg_mask",
                                 type=str,
                                 choices=MASK,
                                 default="none",
                                 help="whether to use segmetation mask")
        self.parser.add_argument("--MIN_OBJECT_AREA",
                                 type=int,
                                 default=20,
                                 help="size threshold to discard mobile masks"
                                      " set as 0 to disable the size screening"
                                 )
        self.parser.add_argument("--boxify",
                                 action="store_true",
                                 help="reshape masks to bounding boxes")
        ## REMOVE MASKED OBJECTS
        self.parser.add_argument("--prob_to_mask_objects", # mixed datasets
                                 type=float,
                                 default=0.0,
                                 help="probability to remove objects "
                                      "overlapping with mobile masks."
                                      " set 0.0 to disable, set 1.0 to"
                                      " objects with 100%")
        ## OPTIONS for SPECIFIC DATASET PREPROCESSING for NUSCENES DATASETS
        self.parser.add_argument("--nuscenes_version",
                            type=str,
                            default ="v1.0-mini",
                            choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
                            help="nuscenes dataset version")
        self.parser.add_argument("--camera_channels",
                            default =["CAM_FRONT"],
                            nargs="+",
                            help="selectable from CAM_FRONT, CAM_FRONT_LEFT, "
                                 "CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, "
                                 "CAM_BACK_RIGHT")
        self.parser.add_argument("--use_keyframe",
                                 action="store_true",
                                 help="whether to use keyframes "
                                      "there are two categories: "
                                      "1. sample_data frames in 12Hz (default) "
                                      "2. keyframes in 2Hz")
        self.parser.add_argument("--enforce_adj_nonkeyframe",
                                 action="store_true",
                                 help="this option only effective when "
                                      "use_keyframe is set True: "
                                      "if True, adjacent non-keyframes would "
                                      " be paired with the central keyframe "
                                      "for image construction; "
                                      "if False, all adjacent and central "
                                      "frames would be keyframes")
        self.parser.add_argument("--stationary_filter",
                                 action="store_true",
                                 help="set True to filter out "
                                      "non-movable objects including "
                                      "traffic cones, barriers, "
                                      "debris and bicycle racks")
        self.parser.add_argument("--speed_limits",
                            default=[0, np.inf],
                            type=float,
                            nargs="+",
                            help="lower and upper speed limits to screen samples")
        self.parser.add_argument("--use_radar",
                                 help="if set, uses radar data for training",
                                 action="store_true")
        self.parser.add_argument("--use_lidar",
                                 help="if set, uses lidar data for training",
                                 action="store_true")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        self.options.file_dir = file_dir
        self.options.data_path = os.path.abspath(
                os.path.expanduser(self.options.data_path)
                )
        if not os.path.exists(self.options.log_dir):
            os.mkdir(self.options.log_dir)
        return self.options
