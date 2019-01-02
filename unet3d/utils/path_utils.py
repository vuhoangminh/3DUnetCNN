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


def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]


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


def get_model_name(model_name, patch_shape, is_crf, depth_unet=None,
                   n_base_filters_unet=None, loss="weighted",
                   model_dim=3):
    model_temp = model_name
    model_temp = "{}{}d".format(model_name, str(model_dim))
    if "unet" in model_name:
        return "ps-{}_{}_crf-{}_d-{}_nb-{}_loss-{}".format(
            patch_shape, model_temp, str(is_crf),
            str(depth_unet), str(n_base_filters_unet), loss)
    else:
        return "ps-{}_{}_crf-{}_loss-{}".format(
            patch_shape, model_temp, str(is_crf), loss)


def get_short_model_name(model_name, patch_shape, is_crf, depth_unet=None,
                         n_base_filters_unet=None, loss="weighted",
                         model_dim=3):
    model_temp = model_name
    model_temp = "{}{}d".format(model_name, str(model_dim))
    if "unet" in model_name:
        return "ps-{}_{}_crf-{}_d-{}_nb-{}".format(
            patch_shape, model_temp, str(is_crf),
            str(depth_unet), str(n_base_filters_unet))
    else:
        return "ps-{}_{}_crf-{}".format(
            patch_shape, model_temp, str(is_crf))


def get_short_core_name(challenge="brats", year="2018",
                        image_shape="160-192-128", crop="1",
                        is_bias_correction="1", is_denoise="0", is_normalize="z",
                        is_hist_match="0"):
    return "{}_{}_is-{}_crop-{}_bias-{}".format(
        challenge, year, image_shape, str(crop), str(
            is_bias_correction))


def get_finetune_name(challenge="brats", year="2018",
                      image_shape="160-192-128", crop="1",
                      is_bias_correction="1", is_denoise="0", is_normalize="z",
                      is_hist_match="0", is_test="1",
                      depth_unet=4, n_base_filters_unet=16,
                      model_name="unet", patch_shape="128-128-128", is_crf="0",
                      loss="weighted", model_dim=3):

    short_model_name = get_short_model_name(model_name=model_name, patch_shape=patch_shape,
                                            is_crf=is_crf, depth_unet=depth_unet,
                                            n_base_filters_unet=n_base_filters_unet,
                                            loss=loss, model_dim=model_dim)

    short_core_name = get_short_core_name(challenge=challenge, year=year,
                                          image_shape=image_shape, crop=crop,
                                          is_bias_correction=is_bias_correction,
                                          is_denoise=is_denoise,
                                          is_normalize=is_normalize,
                                          is_hist_match=is_hist_match)

    return short_model_name, short_core_name


def get_model_baseline_path(folder,
                            challenge="brats", year="2018",
                            image_shape="160-192-128", crop="1",
                            is_bias_correction="1", is_denoise="0", is_normalize="z",
                            is_hist_match="0", is_test="1",
                            depth_unet=4, n_base_filters_unet=16,
                            model_name="unet", patch_shape="128-128-128", is_crf="0",
                            loss="weighted", model_dim=3):
    import glob
    short_model_name, short_core_name = get_finetune_name(challenge=challenge,
                                                          year=year,
                                                          image_shape=image_shape,
                                                          crop=crop,
                                                          is_bias_correction=is_bias_correction,
                                                          is_denoise=is_denoise,
                                                          is_normalize=is_normalize,
                                                          is_hist_match=is_hist_match,
                                                          is_test=is_test,
                                                          depth_unet=depth_unet,
                                                          n_base_filters_unet=n_base_filters_unet,
                                                          model_name=model_name,
                                                          patch_shape=patch_shape,
                                                          is_crf=is_crf,
                                                          loss=loss,
                                                          model_dim=model_dim)
    model_baseline_path = None
    for filename in glob.glob(folder+"/*"):
        print(filename)
        if short_model_name in filename and short_core_name in filename:
            model_baseline_path = filename
    return model_baseline_path


def get_model_h5_filename(datatype, challenge="brats", year="2018",
                          image_shape="160-192-128", crop="1",
                          is_bias_correction="1", is_denoise="0", is_normalize="z",
                          is_hist_match="0", is_test="1",
                          depth_unet=4, n_base_filters_unet=16,
                          model_name="unet", patch_shape="128-128-128", is_crf="0",
                          loss="weighted", model_dim=3):
    core_name = get_core_name(challenge=challenge, year=year,
                              image_shape=image_shape, crop=crop,
                              is_bias_correction=is_bias_correction,
                              is_denoise=is_denoise,
                              is_normalize=is_normalize,
                              is_hist_match=is_hist_match)
    model_full_name = get_model_name(model_name, patch_shape, is_crf,
                                     depth_unet=depth_unet,
                                     n_base_filters_unet=n_base_filters_unet,
                                     loss=loss,
                                     model_dim=model_dim)

    if str2bool(is_test):
        return "test_{}_{}_{}.h5".format(
            core_name, model_full_name, datatype)

    else:
        return "{}_{}_{}.h5".format(
            core_name, model_full_name, datatype)


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


def get_training_h5_paths(brats_dir, overwrite=True, crop=True, challenge="brats", year=2018,
                          image_shape="160-160-128", is_bias_correction="1",
                          is_normalize="z", is_denoise="0",
                          is_hist_match="0", is_test="1",
                          depth_unet=4, n_base_filters_unet=16, model_name="unet",
                          patch_shape="128-128-128", is_crf="0",
                          loss="weighted",
                          is_finetune=False,
                          dir_read_write="base",
                          model_dim=3):

    data_dir = get_h5_training_dir(brats_dir, "data")
    make_dir(data_dir)
    trainids_dir = get_h5_training_dir(brats_dir, "train_val_test_ids")
    make_dir(trainids_dir)
    validids_dir = get_h5_training_dir(brats_dir, "train_val_test_ids")
    make_dir(validids_dir)
    testids_dir = get_h5_training_dir(brats_dir, "train_val_test_ids")
    make_dir(testids_dir)
    model_dir = get_h5_training_dir(brats_dir, "model")
    make_dir(model_dir)

    data_filename = get_training_h5_filename(datatype="data", challenge=challenge,
                                             image_shape=image_shape, crop=crop,
                                             is_bias_correction=is_bias_correction,
                                             is_denoise=is_denoise,
                                             is_normalize=is_normalize,
                                             is_hist_match=is_hist_match,
                                             is_test=is_test)
    model_filename = get_model_h5_filename(datatype="model", challenge=challenge,
                                           image_shape=image_shape, crop=crop,
                                           is_bias_correction=is_bias_correction,
                                           is_denoise=is_denoise,
                                           is_normalize=is_normalize,
                                           is_hist_match=is_hist_match,
                                           depth_unet=depth_unet, n_base_filters_unet=n_base_filters_unet,
                                           model_name=model_name, patch_shape=patch_shape, is_crf=is_crf,
                                           is_test=is_test, loss=loss,
                                           model_dim=model_dim)
    if is_test == "1":
        trainids_filename = "test_train_ids.h5"
        validids_filename = "test_valid_ids.h5"
        testids_filename = "test_test_ids.h5"
    else:
        trainids_filename = "train_ids.h5"
        validids_filename = "valid_ids.h5"
        testids_filename = "test_ids.h5"

    data_path = os.path.join(data_dir, data_filename)
    if is_finetune:
        model_path = os.path.join(model_dir, dir_read_write, model_filename)
    else:
        model_path = os.path.join(model_dir, model_filename)
    trainids_path = os.path.join(trainids_dir, trainids_filename)
    validids_path = os.path.join(validids_dir, validids_filename)
    testids_path = os.path.join(testids_dir, testids_filename)

    return data_path, trainids_path, validids_path, testids_path, model_path


def get_training_h5_paths_old(brats_dir, overwrite=True, crop=True, challenge="brats", year=2018,
                              image_shape="160-160-128", is_bias_correction="1",
                              is_normalize="z", is_denoise="0",
                              is_hist_match="0", is_test="1",
                              depth_unet=4, n_base_filters_unet=16, model_name="unet",
                              patch_shape="128-128-128", is_crf="0",
                              loss="weighted", dir_read_write="base"):

    data_dir = get_h5_training_dir(brats_dir, "data")
    make_dir(data_dir)
    trainids_dir = get_h5_training_dir(brats_dir, "train_ids")
    make_dir(trainids_dir)
    validids_dir = get_h5_training_dir(brats_dir, "valid_ids")
    make_dir(validids_dir)
    testids_dir = get_h5_training_dir(brats_dir, "test_ids")
    make_dir(testids_dir)
    model_dir = get_h5_training_dir(brats_dir, "model")
    make_dir(model_dir)

    data_filename = get_training_h5_filename(datatype="data", challenge=challenge,
                                             image_shape=image_shape, crop=crop,
                                             is_bias_correction=is_bias_correction,
                                             is_denoise=is_denoise,
                                             is_normalize=is_normalize,
                                             is_hist_match=is_hist_match,
                                             is_test=is_test)
    trainids_filename = get_training_h5_filename(datatype="train_ids", challenge=challenge,
                                                 image_shape=image_shape, crop=crop,
                                                 is_bias_correction=is_bias_correction,
                                                 is_denoise=is_denoise,
                                                 is_normalize=is_normalize,
                                                 is_hist_match=is_hist_match,
                                                 is_test=is_test)
    validids_filename = get_training_h5_filename(datatype="valid_ids", challenge=challenge,
                                                 image_shape=image_shape, crop=crop,
                                                 is_bias_correction=is_bias_correction,
                                                 is_denoise=is_denoise,
                                                 is_normalize=is_normalize,
                                                 is_hist_match=is_hist_match,
                                                 is_test=is_test)
    testids_filename = get_training_h5_filename(datatype="test_ids", challenge=challenge,
                                                image_shape=image_shape, crop=crop,
                                                is_bias_correction=is_bias_correction,
                                                is_denoise=is_denoise,
                                                is_normalize=is_normalize,
                                                is_hist_match=is_hist_match,
                                                is_test=is_test)
    model_filename = get_model_h5_filename(datatype="model", challenge=challenge,
                                           image_shape=image_shape, crop=crop,
                                           is_bias_correction=is_bias_correction,
                                           is_denoise=is_denoise,
                                           is_normalize=is_normalize,
                                           is_hist_match=is_hist_match,
                                           depth_unet=depth_unet, n_base_filters_unet=n_base_filters_unet,
                                           model_name=model_name, patch_shape=patch_shape, is_crf=is_crf,
                                           is_test=is_test, loss=loss)

    data_path = os.path.join(data_dir, data_filename)
    trainids_path = os.path.join(trainids_dir, trainids_filename)
    validids_path = os.path.join(validids_dir, validids_filename)
    testids_path = os.path.join(testids_dir, testids_filename)
    model_path = os.path.join(model_dir, dir_read_write, model_filename)

    return data_path, trainids_path, validids_path, testids_path, model_path
