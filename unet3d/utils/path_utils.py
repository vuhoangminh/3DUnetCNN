# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04
Copyright (c) 2018, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

import os
import ntpath
from unet3d.utils.print_utils import print_processing, print_section, print_separator

def get_project_dir(path, project_name):
    paths = path.split(project_name)
    return paths[0] + project_name


def split_dos_path_into_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    folders.reverse()
    return folders


def get_h5_image_path(brats_dir,
                      is_normalize_mean_std=False,
                      challenge=2018,
                      dataset="original"):
    if is_normalize_mean_std:
        dataset_fullname = "brats{}_{}_normalize_mean_std".format(
            challenge, dataset)
    else:
        dataset_fullname = "brats{}_{}_normalize_minh".format(
            challenge, dataset)

    save_to_dir = os.path.join(brats_dir, "database", dataset_fullname)
    return save_to_dir


def get_data_dir(brats_dir, data_folder="data_train", dataset="test"):
    return os.path.join(brats_dir, data_folder, dataset)


def get_analysis_dir(dataset_dir, data_folder):
    return os.path.join(dataset_dir, "database", data_folder)


def get_normalize_minh_dir(brats_dir, data_folder="data_train", dataset="test"):
    return os.path.join(brats_dir, data_folder, dataset + "_minh_normalize")


def get_normalize_minh_file_path(path, dataset="test"):
    return path.replace(dataset, dataset + "_minh_normalize")


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_modality(path, ext=".nii.gz"):
    filename = get_filename(path)
    modality = filename.replace(ext, "")
    return modality


def make_dir(dir):
    if not os.path.exists(dir):
        print_separator()
        print("making dir", dir)
        os.makedirs(dir)


def get_template_path(path, brats_dir, dataset="test", template_data_folder="data_train", template_folder="HGG/Brats18_2013_2_1"):
    filename = get_filename(path)
    template_path = os.path.join(brats_dir, template_data_folder,
                                 dataset, template_folder,
                                 filename)
    return template_path        