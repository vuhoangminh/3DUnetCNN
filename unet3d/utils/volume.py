import os
import math
import ntpath
import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random
import itertools
from brats.correct_name import config

def get_shape(volume):
    return volume.shape


def get_bounding_box_nd(volume):
    N = volume.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(volume, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


def get_bounding_box(volume):
    r = np.any(volume, axis=(1, 2))
    c = np.any(volume, axis=(0, 2))
    z = np.any(volume, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return np.array([rmin.T, rmax.T, cmin.T, cmax.T, zmin.T, zmax.T])


def get_size_bounding_box(volume):
    rmin, rmax, cmin, cmax, zmin, zmax = get_bounding_box(volume)
    return np.array([rmax - rmin, cmax-cmin, zmax-zmin])


def get_non_zeros_pixel(volume):
    return np.count_nonzero(volume)


def get_zeros_pixel(volume):
    return volume.size - np.count_nonzero(volume)


def compute_mean_non_zeros_pixel(volume):
    return volume[volume != 0].mean()


def compute_std_non_zeros_pixel(volume):
    return np.nanstd(np.where(np.isclose(volume, 0), np.nan, volume))


def count_number_occurrences_label(truth):
    temp_truth = truth.astype(int)
    truth_reshape = temp_truth.ravel()
    return np.bincount(truth_reshape)


def get_unique_label(truth):
    temp_truth = truth.astype(int)
    truth_reshape = temp_truth.ravel()
    return np.unique(truth_reshape)


def get_max_min_intensity(volume):
    return np.max(volume), np.min(volume), np.min(volume[volume != 0])


def get_size(volume_path):
    return round(os.path.getsize(volume_path)/1000000, 1)


def count_non_zeros_background(volume, truth):
    indice_volume = volume > 0
    indice_truth = truth == 0
    indice = np.multiply(indice_volume, indice_truth)
    return np.count_nonzero(indice)


def count_zeros_non_background(volume, truth):
    indice_volume = volume <= 0
    indice_truth = truth != 0
    indice = np.multiply(indice_volume, indice_truth)
    return np.count_nonzero(indice)


def get_filename_with_extenstion(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_filename_without_extenstion(path):
    filename = get_filename_with_extenstion(path)
    return filename.replace(".nii.gz", "")


def get_truth_path(volume_path):
    volume_filename = get_filename_without_extenstion(volume_path)
    truth_path = volume_path.replace(volume_filename, config["truth"][0])
    return truth_path


def get_volume_paths(truth_path):
    volume_paths = list()
    for modality in config["training_modalities"]:
        volume_path = truth_path.replace(config["truth"][0], modality)
        volume_paths.append(volume_path)
    return volume_paths