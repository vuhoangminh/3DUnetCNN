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


def get_data_dir(brats_dir, data_folder="data_train", dataset="tets"):
    return os.path.join(brats_dir, data_folder, dataset)


def get_analysis_dir(dataset_dir, data_folder):
    return os.path.join(dataset_dir, "database", data_folder)


def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_modality(path, ext=".nii.gz"):
    filename = get_filename(path)
    modality = filename.replace(ext, "")
    return modality