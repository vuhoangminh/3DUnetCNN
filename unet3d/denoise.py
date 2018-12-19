import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def denoise_data(data, is_denoise="gaussian"):

    for i in range(data.shape[0]):
        volume = data[i, :, :, :]
        if is_denoise == "gaussian":
            volume_denoise = gaussian_filter(volume, sigma=0.5)
        else:
            raise Exception("{} not support", is_denoise)

        data[i, :, :, :] = volume_denoise

        # save_nib(template, temp_template_path, affine)
        # save_nib(volume_normalized, temp_volume_norm_path, affine)
        # save_nib(volume_normalized-volume, temp_diff_path, affine)
    return data


def denoise_data_storage(data_storage, is_denoise="gaussian"):
    for index in range(data_storage.shape[0]):
        data_storage[index] = denoise_data(data_storage[index],
                                           is_denoise=is_denoise)
    return data_storage