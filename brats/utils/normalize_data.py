# -*- coding: utf-8 -*-
"""
Created on Wed Dec 05
Copyright (c) 2018, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

import os
import glob
import argparse
import shutil
import nibabel as nib

from brats.config import config
from unet3d.normalize_minh import hist_match_non_zeros
from unet3d.utils.path_utils import get_project_dir, get_h5_image_path, get_parent_dir, make_dir
from unet3d.utils.path_utils import get_data_dir, get_normalize_minh_dir, get_normalize_minh_file_path
from unet3d.utils.path_utils import get_template_path
from unet3d.utils.print_utils import print_processing, print_section, print_separator


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def get_args():
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=config["dataset"],
                        default="test",
                        help="dataset type")
    parser.add_argument('-o', '--overwrite', type=bool,
                        default=False)
    parser.add_argument('-f', '--data_folder', type=str,
                        choices=config["data_folders"],
                        default="data_train",
                        help="data folders")
    args = parser.parse_args()
    return args


def normalize_one_folder(data_folder, dataset, overwrite=False):
    normalize_minh_dir = get_normalize_minh_dir(brats_dir=BRATS_DIR,
                                                data_folder=data_folder,
                                                dataset=dataset)

    make_dir(normalize_minh_dir)

    data_dir = get_data_dir(brats_dir=BRATS_DIR,
                            data_folder=data_folder, dataset=dataset)
    subject_paths = glob.glob(os.path.join(data_dir, "*", "*", "*.nii.gz"))

    for i in range(len(subject_paths)):
        subject_path = subject_paths[i]
        normalize_minh_file_path = get_normalize_minh_file_path(path=subject_path, dataset=dataset)
        parent_dir = get_parent_dir(normalize_minh_file_path)
        make_dir(parent_dir)

        print_processing(subject_path)

        template_path = get_template_path(
            path=subject_path, dataset=dataset, brats_dir=BRATS_DIR,
            template_data_folder=config["template_data_folder"],
            template_folder=config["template_folder"])

        template = nib.load(template_path)
        template = template.get_fdata()

        if overwrite or not os.path.exists(normalize_minh_file_path):
            if config["truth"][0] in normalize_minh_file_path:
                print("saving truth to", normalize_minh_file_path)
                shutil.copy(subject_path, normalize_minh_file_path)
            elif config["mask"][0] in normalize_minh_file_path:
                print("saving mask to", normalize_minh_file_path)
                shutil.copy(subject_path, normalize_minh_file_path)
            else:
                volume = nib.load(subject_path)
                affine = volume.affine
                volume = volume.get_fdata()
                source_hist_match = hist_match_non_zeros(volume, template)
                print("saving to", normalize_minh_file_path)
                source_hist_match = nib.Nifti1Image(source_hist_match, affine=affine)
                nib.save(source_hist_match, normalize_minh_file_path)



def main(overwrite=False):
    args = get_args()
    dataset = args.dataset
    overwrite = args.overwrite
    data_folder = args.data_folder

    normalize_one_folder(data_folder=data_folder,
                         dataset=dataset, overwrite=overwrite)


if __name__ == "__main__":
    main(False)
