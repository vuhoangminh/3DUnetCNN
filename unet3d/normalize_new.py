import os

import numpy as np
import nibabel as nib

from nilearn.masking import compute_multi_background_mask
from nilearn.image import new_img_like
from skimage import exposure

from .utils import crop_img, crop_img_to, read_image

from unet3d.utils.utils import resize, read_image_files
from unet3d.utils.path_utils import get_filename, get_template_path, get_mask_path_from_set_of_files

from unet3d.utils.volume import get_background_mask
from brats.config import config

from unet3d.utils.utils import str2bool


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape,
                         interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters_old(in_files):
    if len(in_files) > 1:
        foreground = get_complete_foreground(in_files)
    else:
        foreground = get_foreground_from_set_of_files(
            in_files[0], return_image=True)
    return crop_img(foreground, return_slices=True, copy=True)


def get_cropping_parameters(in_files):
    mask = get_background_mask(in_files[0][0])
    return crop_img(mask, return_slices=True, copy=True)


def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False):
    if crop:
        crop_slices = get_cropping_parameters([in_files])
    else:
        crop_slices = None
    images = read_image_files(
        in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices)
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images


def get_complete_foreground(training_data_files):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, return_image=False):
    volumes_data = list()
    for path in set_of_files:
        volume = read_image(path)
        volumes_data.append(volume)
    background_image = compute_multi_background_mask(volumes_data)
    if return_image:
        return new_img_like(volume, background_image)
    else:
        return background_image


def normalize_z(data):
    data_norm = data - np.mean(data)
    data_norm = data_norm/np.std(data)
    return data_norm


def normalize_01(data):
    x_min = np.min(data)
    x_max = np.max(data)
    data_norm = (data - x_min)/(x_max-x_min)
    return data_norm


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def normalize_volume(source, template,
                     is_normalize="z",
                     is_hist_match="0"):
    # set negative to 0
    source[source < 0] = 0
    template[template < 0] = 0

    # reshape to 1d
    source_1d = source.reshape((source.size))
    template_1d = template.reshape((template.size))

    # extract index
    idx_source = np.argwhere(source_1d > 0)
    idx_template = np.argwhere(template_1d > 0)

    # get array
    source_norm_array = np.ndarray.take(source, idx_source)
    template_norm_array = np.take(template, idx_template)

    # normalize to 0-1
    if is_normalize == "01":
        source_norm_array = normalize_01(source_norm_array)
        template_norm_array = normalize_01(template_norm_array)
    elif is_normalize == "z":
        source_norm_array = normalize_z(source_norm_array)
        template_norm_array = normalize_z(template_norm_array)

    # hist matching")
    if str2bool(is_hist_match):
        source_hist_match_array = hist_match(
            source_norm_array, template_norm_array)
    else:
        source_hist_match_array = source_norm_array

    # copy data back
    source_norm = np.zeros((source.size, 1))
    source_norm[idx_source.flatten()] = source_hist_match_array

    # reshape to original shape
    source_norm = source_norm.reshape(source.shape)

    return source_norm


def normalize_data(data, data_paths, brats_dir,
                   dataset="test",
                   is_normalize="z",
                   is_hist_match="0"):

    for i in range(data.shape[0]):
        volume = data[i, :, :, :]
        data_path = data_paths[i]
        template_path = get_template_path(
            path=data_path, dataset=dataset, brats_dir=brats_dir,
            template_data_folder=config["template_data_folder"],
            template_folder=config["template_folder"])

        template = nib.load(template_path)

        # affine = template.affine
        # save_nib(volume, temp_volume_path, affine)

        template = template.get_data()
        volume_normalized = normalize_volume(volume, template,
                                             is_normalize=is_normalize,
                                             is_hist_match=is_hist_match)
        data[i, :, :, :] = volume_normalized

        # save_nib(template, temp_template_path, affine)
        # save_nib(volume_normalized, temp_volume_norm_path, affine)
        # save_nib(volume_normalized-volume, temp_diff_path, affine)
    return data


def normalize_data_storage(data_storage, training_data_files, brats_dir,
                           dataset="original", is_normalize="z", is_hist_match="0"):
    for index in range(data_storage.shape[0]):
        data_paths = training_data_files[index]
        data_storage[index] = normalize_data(data_storage[index],
                                             data_paths,
                                             brats_dir,
                                             dataset=dataset,
                                             is_normalize=is_normalize,
                                             is_hist_match=is_hist_match)
    return data_storage
