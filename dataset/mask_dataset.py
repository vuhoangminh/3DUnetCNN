# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04
Copyright (c) 2018, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

import os
import glob
import shutil
import sys
import nibabel as nib
import argparse

from unet3d.utils.path_utils import get_project_dir, get_data_dir
from unet3d.utils.path_utils import split_dos_path_into_components
from unet3d.utils.path_utils import get_modality
from unet3d.utils.print_utils import print_processing, print_section, print_separator
from unet3d.utils.volume import get_background_mask, get_volume_paths_from_one_volume


from brats.config import config
from brats.prepare_data import get_h5_image_path

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def get_mask_path(volume_path):
    modality = get_modality(volume_path)
    return volume_path.replace(modality, config["mask"])


def get_args():
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=config["dataset"],
                        default="test",
                        help="dataset type")
    parser.add_argument('-f', '--data_folder', type=str,
                        choices=config["data_folders"],
                        default="data_train",
                        help="data folders")
    parser.add_argument('-o', '--overwrite', type=bool,
                        default=True)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset = args.dataset
    data_folder = args.data_folder
    overwrite = args.overwrite

    data_dir = get_data_dir(brats_dir=BRATS_DIR,
                            data_folder=data_folder, dataset=dataset)
    subject_dirs = glob.glob(os.path.join(data_dir, "*", "*", "*.nii.gz"))

    list_processed = list()
    for i in range(len(subject_dirs)):
        subject_dir = subject_dirs[i]
        if subject_dir not in list_processed:
            volume_paths = get_volume_paths_from_one_volume(subject_dir)
            list_processed.extend(volume_paths)
            if config["mask"] not in subject_dir and config["truth"][0] not in subject_dir:
                print_processing(subject_dir)
                mask_path = get_mask_path(subject_dir)
                if overwrite or not os.path.exists(mask_path):
                    mask = get_background_mask(subject_dir)
                else:
                    print("mask exists")
                nib.save(mask, mask_path)


if __name__ == "__main__":
    main()
