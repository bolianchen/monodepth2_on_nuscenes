# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
from PIL import Image
import numpy as np
import cv2


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def image_resize(image, target_h, target_w, shift_h, shift_w,
                 inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # get the raw image size

    is_pil = isinstance(image, Image.Image)

    if is_pil:
        image = np.array(image)

    (raw_h, raw_w) = image.shape[:2]

    assert raw_h >= target_h, 'must be downscaling'
    assert raw_w >= target_w, 'must be downscaling'

    if target_h/raw_h <= target_w/raw_w:
        # calculate the ratio of the width and construct the dimensions
        r = target_w / float(raw_w)
        dim = (target_w, int(raw_h * r))

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]
        
        start = int(new_h*shift_h)
        end = start + target_h
       
        assert start >=0
        assert end <= new_h

        if len(image.shape) == 3:
            image = image[start:end,:,:]
        else:
            image = image[start:end,:]

        delta_u = 0
        delta_v = start  

    else: 
        # calculate the ratio of the height and construct the dimensions
        r = target_h / float(raw_h)
        dim = (int(raw_w * r), target_h)

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]

        start = int(new_w*shift_w)
        end = start + target_w
        image = image[:,start:end,:]

        assert start >=0
        assert end <= new_w

        if len(image.shape) == 3:
            image = image[:,start:end,:]
        else:
            image = image[:,start:end]

        delta_u = start
        delta_v = 0

    if is_pil:
        image = Image.fromarray(image)

    return image, r, delta_u, delta_v

def check_if_scene_pass(scene_description, pass_filters=['day', 'night', 'rain']):
    """Return True if scene_description meets the specified pass_filter
    Args:
        scene_description(str): description included in the scene metadata
        pass_filters(list of str): check the consequeneces below
            ['day']: daytime and not rainy scenes
            ['night']: nighttime and not rainy scenes
            ['rain']: rainy scenes on both daytime and nighttime
            ['day', 'night']: daytime, nighttime, and not rainy scenes
            ['day', 'rain']: rainy scenes on daytime
            ['night', 'rain']: rainy scenes on nighttime
            ['day', 'night', 'rain']: all the scenes
    """
    description = scene_description.lower()
    pass_filters = [fil for fil in pass_filters
            if fil in ('day', 'night', 'rain')]

    if len(pass_filters) == 0:
        raise NameError('invalid options are detected in pass_filters')

    if len(pass_filters) == 3:
        return True

    if 'day' in pass_filters:
        # ['day', 'night']
        if 'night' in pass_filters:
            if 'rain' not in description:
                return True
        # ['day', 'rain']
        elif 'rain' in pass_filters:
            if 'night' not in description and 'rain' in description:
                return True
        # ['day']
        else:
            if 'night' not in description and 'rain' not in description:
                return True

    if 'night' in pass_filters:
        # ['night', 'rain']
        if 'rain' in pass_filters:
            if 'night' in description and 'rain' in description:
                return True
        # ['night']
        else:
            if 'night' in description and 'rain' not in description:
                return True

    # ['rain']
    if pass_filters == ['rain']:
        if 'rain' in description:
            return True

    return False
