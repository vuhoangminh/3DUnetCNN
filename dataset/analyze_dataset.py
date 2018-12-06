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
import nibabel as nib
import argparse
import pandas as pd

from unet3d.utils.path_utils import get_project_dir, get_h5_image_path, get_data_dir, get_analysis_dir
from unet3d.utils.path_utils import split_dos_path_into_components
from unet3d.utils.print_utils import print_processing, print_section, print_separator
from unet3d.utils.volume import get_truth_path, get_volume_paths, is_truth_path
from unet3d.utils.volume import get_size
from unet3d.utils.volume import count_non_zeros_background, count_zeros_non_background
from unet3d.utils.volume import get_max_min_intensity
from unet3d.utils.volume import get_unique_label
from unet3d.utils.volume import count_number_occurrences_label
from unet3d.utils.volume import compute_mean_non_zeros_pixel, compute_std_non_zeros_pixel
from unet3d.utils.volume import get_non_zeros_pixel, get_zeros_pixel
from unet3d.utils.volume import get_shape
from unet3d.utils.volume import get_size_bounding_box
from unet3d.utils.volume import get_bounding_box

from brats.config import config

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])

columns = ["dataset",
           "folder",
           "name",
           "modality",
           "size",
           "shape",
           "bounding_box",
           "size_bounding_box",
           "n_non_zeros_pixel",
           "n_zeros_pixel",
           "mean_non_zeros_pixel",
           "std_non_zeros_pixel",
           "n_occurrences_label",
           "n_unique_label",
           "max_intensity",
           "min_intensity",
           "min_intensity_non_zeros",
           "n_non_zeros_background",
           "n_zeros_non_background"
           ]


def get_header_info(path):
    folders = split_dos_path_into_components(path)
    N = len(folders)
    dataset = folders[N-4]
    folder = folders[N-3]
    name = folders[N-2]
    modality = folders[N-1].replace(".nii.gz", "")
    return dataset, folder, name, modality





def analyze_one_folder(data_folder, dataset, overwrite=False):

    analysis_dir = get_analysis_dir(DATASET_DIR, data_folder)
    analysis_file_path = os.path.join(analysis_dir, dataset + ".xlsx")
    print("save to dir", analysis_dir)
    print("save to file", analysis_file_path)

    if not os.path.exists(analysis_dir):
        print_separator()
        print("making dir", analysis_dir)
        os.makedirs(analysis_dir)

    if overwrite or not os.path.exists(analysis_file_path):
        writer = pd.ExcelWriter(analysis_file_path)

        data_dir = get_data_dir(brats_dir=BRATS_DIR,
                                data_folder=data_folder, dataset=dataset)
        subject_dirs = glob.glob(os.path.join(data_dir, "*", "*", "*.nii.gz"))

        index = range(0, len(subject_dirs)-1, 1)
        df = pd.DataFrame(index=index, columns=columns)

        for i in range(len(subject_dirs)):
            subject_dir = subject_dirs[i]
            print_processing(subject_dir)

            dataset, folder, name, modality = get_header_info(subject_dir)

            volume = nib.load(subject_dir)
            volume = volume.get_fdata()

            df["dataset"][i] = dataset
            df["folder"][i] = folder
            df["name"][i] = name
            df["modality"][i] = modality
            df["size"][i] = get_size(subject_dir)
            df["shape"][i] = get_shape(volume)
            df["bounding_box"][i] = get_bounding_box(volume)
            df["size_bounding_box"][i] = get_size_bounding_box(volume)
            df["n_non_zeros_pixel"][i] = get_non_zeros_pixel(volume)
            df["n_zeros_pixel"][i] = get_zeros_pixel(volume)
            df["mean_non_zeros_pixel"][i] = compute_mean_non_zeros_pixel(
                volume)
            df["std_non_zeros_pixel"][i] = compute_std_non_zeros_pixel(volume)
            df["max_intensity"][i], df["min_intensity"][i], df["min_intensity_non_zeros"][i] = get_max_min_intensity(
                volume)

            if "valid" not in data_folder:
                if not is_truth_path(subject_dir):
                    truth_path = get_truth_path(subject_dir)
                    truth = nib.load(truth_path)
                    truth = truth.get_fdata()
                    df["n_non_zeros_background"][i] = count_non_zeros_background(
                        volume, truth)
                    df["n_zeros_non_background"][i] = count_zeros_non_background(
                        volume, truth)
                else:
                    df["n_occurrences_label"][i] = count_number_occurrences_label(
                        volume)
                    df["n_unique_label"][i] = get_unique_label(volume)

        df.to_excel(writer, 'Sheet1')
        writer.save()


def get_args():
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=config["dataset"]+config["dataset_minh_normalize"],
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

    print(args)

    analyze_one_folder(data_folder, dataset, overwrite)


if __name__ == "__main__":
    main()
