# -*- coding: utf-8 -*-
"""
Created on Mon Dec 03
Copyright (c) 2018, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

import os
import glob
import shutil
import sys
import argparse
import nibabel as nib
import pandas as pd
import numpy as np

import unet3d.utils.print_utils as print_utils
import unet3d.utils.path_utils as path_utils
import unet3d.utils.args_utils as get_args
from unet3d.utils.volume import get_spacing, get_shape
from unet3d.utils.utils import resize, pad, read_image_files

from nilearn.image import reorder_img, new_img_like


# def find_scaling_dim(dim_in, spacing, thickness=(2.5, 1, 1)):
# def find_scaling_dim(dim_in, spacing, thickness=(1, 1, 2.5)):
#     dim_out_x = int(np.round(dim_in[0]*spacing[0]/thickness[0]))
#     dim_out_y = int(np.round(dim_in[1]*spacing[1]/thickness[1]))
#     dim_out_z = int(np.round(dim_in[2]*spacing[2]/thickness[2]))
#     return (dim_out_x, dim_out_y, dim_out_z)


def find_scaling_dim(dim_in, spacing, thickness=(2.5, 1, 1)):
    dim_out_x = int(np.round(dim_in[0]*spacing[0]/thickness[0]))
    dim_out_y = int(np.round(dim_in[1]*spacing[1]/thickness[1]))
    dim_out_z = int(np.round(dim_in[2]*spacing[2]/thickness[2]))
    return (dim_out_y, dim_out_z, dim_out_x)
    # return (dim_out_x, dim_out_y, dim_out_z)


def rescale_and_pad(out_dir, volume, affine, dim_scaling, dim_out=(576, 576, 300), interpolation="linear"):
    # volume = nib.Nifti1Image(volume, affine)
    volume = resize(volume, new_shape=dim_scaling, interpolation=interpolation)

    # if interpolation == "linear":
    #     volume.to_filename(os.path.join(out_dir, "rescale_volume.nii.gz"))
    # else:
    #     volume.to_filename(os.path.join(out_dir, "rescale_truth.nii.gz"))

    volume = pad(volume, new_shape=dim_out, mode="constant",
                 interpolation=interpolation)

    # if interpolation == "linear":
    #     volume.to_filename(os.path.join(out_dir, "pad_rescale_volume.nii.gz"))
    # else:
    #     volume.to_filename(os.path.join(out_dir, "pad_rescale_truth.nii.gz"))

    return volume


def prerocess(data_folder, dataset, config, overwrite=False):

    data_dir = path_utils.get_analysis_dir(DATASET_DIR, data_folder)
    print("save to dir", data_dir)

    if not os.path.exists(data_dir):
        print_utils.print_separator()
        print("making dir", data_dir)
        os.makedirs(data_dir)

    data_dir = os.path.join(data_dir, dataset)
    out_dir = data_dir.replace(dataset, "preprocessed")

    path_utils.make_dir(out_dir)

    subject_dirs = glob.glob(os.path.join(
        data_dir, "*"))

    for i in range(len(subject_dirs)):
        folder = subject_dirs[i]
        # if "case_00073" not in folder:
        #     continue
        out_dir = folder.replace(dataset, "preprocessed")
        path_utils.make_dir(out_dir)

        volume_path_out = os.path.join(out_dir, "imaging.nii.gz")
        truth_path_out = os.path.join(out_dir, "segmentation.nii.gz")
        if os.path.exists(volume_path_out) and os.path.exists(truth_path_out):
            continue

        print_utils.print_processing(folder)
        volume_path = os.path.join(folder, "imaging.nii.gz")
        truth_path = os.path.join(folder, "segmentation.nii.gz")

        volume = nib.load(volume_path)
        truth = nib.load(truth_path)

        # volume = reorder_img(volume, resample="linear")
        # volume.to_filename(os.path.join(out_dir, "reorder_volume.nii.gz"))

        # truth = reorder_img(truth, resample="nearest")
        # truth.to_filename(os.path.join(out_dir, "reorder_truth.nii.gz"))

        shape = get_shape(volume)
        spacing = get_spacing(volume)
        affine = volume.affine
        # volume = volume.get_fdata()
        # truth = truth.get_fdata()

        dim_scaling = find_scaling_dim(shape, spacing=spacing)

        volume = rescale_and_pad(
            out_dir, volume, affine, dim_scaling=dim_scaling, interpolation="linear")
        truth = rescale_and_pad(
            out_dir, truth, affine, dim_scaling=dim_scaling, interpolation="nearest")

        volume.to_filename(volume_path_out)
        truth.to_filename(truth_path_out)


def main():

    global BRATS_DIR, DATASET_DIR, PROJECT_DIR

    from projects.pros.config import config
    args = get_args.train_pros()

    challenge = args.challenge
    dataset = "original"
    data_folder = "data_train"
    overwrite = True

    CURRENT_WORKING_DIR = os.path.realpath(__file__)
    PROJECT_DIR = path_utils.get_project_dir(
        CURRENT_WORKING_DIR, config["project_name"])
    BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
    DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])

    print(args)

    prerocess(data_folder, dataset,
              config=config, overwrite=overwrite)


if __name__ == "__main__":
    main()
