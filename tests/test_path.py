# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04
Copyright (c) 2018, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

import os
from unet3d.utils.path_utils import get_project_dir
from unet3d.utils.volume import get_volume_paths_from_one_volume

from dataset.mask_dataset import get_mask_path
from brats.config import config

project_name = config["project_name"]
current_working_dir = os.getcwd()

project_path = get_project_dir(current_working_dir, project_name)
brats_path = os.path.join(project_path, config["brats_folder"])
dataset_path = os.path.join(project_path, config["dataset_folder"])


print(project_path)
print(brats_path)
print(dataset_path)


volume_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t1ce.nii.gz"
mask_path = get_mask_path(volume_path)

print(mask_path)

volume_paths = get_volume_paths_from_one_volume(volume_path)
for volume_path in volume_paths:
    print(volume_path)