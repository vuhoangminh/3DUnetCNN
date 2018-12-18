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
import numpy as np
from unet3d.utils.utils import str2bool


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


# brats_2018_is-160-192-128_bias-1_denoise-0_norm-z_ps-128-128-128_unet_crf-0_d-4_nb-16.h5
# brats_2018_is-160-192-128_bias-1_denoise-bm4d_norm-minh_data.h5
# brats_2018_is-160-192-128_bias-1_denoise-bm4d_norm-minh_train_ids.h5
# brats_2018_is-160-192-128_bias-1_denoise-bm4d_norm-minh_valid_ids.h5

def get_h5_training_dir(brats_dir, datatype="data"):
    return os.path.join(brats_dir, "database", datatype)


def get_core_name(challenge="brats", year="2018",
                  image_shape="160-192-128", crop="1",
                  is_bias_correction="1", is_denoise="0", is_normalize="z",
                  is_hist_match="0"):
    return "{}_{}_is-{}_crop-{}_bias-{}_denoise-{}_norm-{}_hist-{}".format(
        challenge, year, image_shape, str(crop), str(
            is_bias_correction), str(is_denoise),
        str(is_normalize), str(is_hist_match))


def get_model_name(model, patch_shape, is_crf, depth_unet=None, n_base_filters_unet=None):
    if model == "unet":
        return "ps-{}_{}_crf-{}_d-{}_nb-{}".format(
            patch_shape, model, str(is_crf),
            str(depth_unet), str(n_base_filters_unet))
    else:
        return "ps-{}_{}_crf-{}".format(
            patch_shape, model, str(is_crf))


def get_model_h5_filename(datatype, challenge="brats", year="2018",
                          image_shape="160-192-128", crop="1",
                          is_bias_correction="1", is_denoise="0", is_normalize="z",
                          is_hist_match="0", is_test="1",
                          depth_unet=4, n_base_filters_unet=16,
                          model="unet", patch_shape="128-128-128", is_crf="0"):
    core_name = get_core_name(challenge=challenge, year=year,
                              image_shape=image_shape, crop=crop,
                              is_bias_correction=is_bias_correction,
                              is_denoise=is_denoise,
                              is_normalize=is_normalize,
                              is_hist_match=is_hist_match)
    model_name = get_model_name(model, patch_shape, is_crf,
                                depth_unet=depth_unet, n_base_filters_unet=n_base_filters_unet)

    if str2bool(is_test):
        return "test_{}_{}_{}.h5".format(
            core_name, model_name, datatype)

    else:
        return "{}_{}_{}.h5".format(
            core_name, model_name, datatype)


def get_training_h5_filename(datatype, challenge="brats", year="2018",
                             image_shape="160-192-128", crop="1",
                             is_bias_correction="1", is_denoise="0", is_normalize="z",
                             is_hist_match="0", is_test="1"):
    core_name = get_core_name(challenge=challenge, year=year,
                              image_shape=image_shape, crop=crop,
                              is_bias_correction=is_bias_correction,
                              is_denoise=is_denoise,
                              is_normalize=is_normalize,
                              is_hist_match=is_hist_match)
    if str2bool(is_test):
        return "test_{}_{}.h5".format(core_name, datatype)
    else:
        return "{}_{}.h5".format(core_name, datatype)


def get_mask_path_from_set_of_files(in_files):
    for file in in_files:
        if "mask.nii.gz" in file:
            return file


def get_shape_string(image_shape):
    shape_string = ""
    for i in range(len(image_shape)-1):
        shape_string = shape_string + str(image_shape[i]) + "-"
    shape_string = shape_string + str(image_shape[-1])
    return shape_string


def get_shape_from_string(shape_string):
    splitted_string = shape_string.split("-")
    splitted_number = list(map(int, splitted_string))
    return tuple(splitted_number)
