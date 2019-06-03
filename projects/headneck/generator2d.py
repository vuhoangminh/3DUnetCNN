import os
import copy
from random import shuffle
import itertools

import numpy as np
import time

from unet3d.utils import pickle_dump, pickle_load
from unet3d.generator import get_train_valid_test_split, get_number_of_steps
from unet3d.generator import get_train_valid_test_split_isbr
from unet3d.generator import get_number_of_patches, create_patch_index_list
from unet3d.generator import get_multi_class_labels, get_data_from_file
from unet3d.utils.threadsafe import threadsafe_generator

import tensorlayer as tl
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

import unet3d.utils.image_utils as image_utils

# from unet3d.generator import get_training_and_validation_and_testing_generators


def get_training_and_validation_and_testing_generators2d(data_file, batch_size, n_labels, training_keys_file,
                                                         validation_keys_file, testing_keys_file,
                                                         data_split=0.8, overwrite=False, labels=None, patch_shape=None,
                                                         validation_patch_overlap=0, training_patch_start_offset=None,
                                                         validation_batch_size=None,
                                                         augment_flipud=False, augment_fliplr=False, augment_elastic=False,
                                                         augment_rotation=False, augment_shift=False, augment_shear=False,
                                                         augment_zoom=False, n_augment=0, skip_blank=False, is_test="1",
                                                         patch_overlap=[
                                                             0, 0, -1],
                                                         project="brats",
                                                         is_extract_patch_agressive=False,
                                                         data_type_generator="combined"):
    """
    Creates the training and validation generators that can be used when training the model.
    :param skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
    :param validation_batch_size: Batch size for the validation data.
    :param training_patch_start_offset: Tuple of length 3 containing integer values. Training data will randomly be
    offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
    :param validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
    patch_shape to not be None)
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :param permute: will randomly permute the data (data must be 3D cube)
    :param data_type_generator: "combined", "cascaded", "separated"
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """

    if not validation_batch_size:
        validation_batch_size = batch_size

    if project == "brats":
        training_list, validation_list, _ = get_train_valid_test_split(
            data_file, training_file=training_keys_file,
            validation_file=validation_keys_file,
            testing_file=testing_keys_file,
            data_split=data_split, overwrite=False)
    else:
        training_list, validation_list, _ = get_train_valid_test_split_isbr(
            data_file, training_file=training_keys_file,
            validation_file=validation_keys_file,
            testing_file=testing_keys_file,
            overwrite=False)

    print("training_list:", training_list)

    print(">> training data generator")
    patch_overlap = np.asarray(patch_overlap)
    training_generator = data_generator2d(data_file, training_list,
                                          batch_size=batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          patch_shape=patch_shape,
                                          patch_overlap=patch_overlap,
                                          patch_start_offset=training_patch_start_offset,
                                          augment_flipud=augment_flipud,
                                          augment_fliplr=augment_fliplr,
                                          augment_elastic=augment_elastic,
                                          augment_rotation=augment_rotation,
                                          augment_shift=augment_shift,
                                          augment_shear=augment_shear,
                                          augment_zoom=augment_zoom,
                                          n_augment=n_augment,
                                          skip_blank=skip_blank,
                                          is_extract_patch_agressive=is_extract_patch_agressive,
                                          data_type_generator=data_type_generator)
    print(">> valid data generator")
    validation_generator = data_generator2d(data_file, validation_list,
                                            batch_size=validation_batch_size,
                                            n_labels=n_labels,
                                            labels=labels,
                                            patch_shape=patch_shape,
                                            patch_overlap=0,
                                            skip_blank=skip_blank,
                                            augment_flipud=augment_flipud,
                                            augment_fliplr=augment_fliplr,
                                            augment_elastic=augment_elastic,
                                            augment_rotation=augment_rotation,
                                            augment_shift=augment_shift,
                                            augment_shear=augment_shear,
                                            augment_zoom=augment_zoom,
                                            n_augment=n_augment,
                                            is_extract_patch_agressive=is_extract_patch_agressive,
                                            data_type_generator=data_type_generator)

    # Set the number of training and testing samples per epoch correctly
    print(">> compute number of training and validation steps")

    # num_training_steps = get_number_of_steps(get_number_of_patches2d(data_file, training_list, patch_shape,
    #                                                                  patch_start_offset=training_patch_start_offset,
    #                                                                  patch_overlap=patch_overlap,
    #                                                                  data_type_generator=data_type_generator),
    #                                          batch_size)
    # num_validation_steps = get_number_of_steps(get_number_of_patches2d(data_file, validation_list, patch_shape,
    #                                                                    patch_overlap=0,
    #                                                                    data_type_generator=data_type_generator),
    #                                            validation_batch_size)

    # else:
    #     # num_training_steps = get_number_of_steps(11137, batch_size)

    # 1-512-512
    # num_training_steps = get_number_of_steps(6690, batch_size)
    # num_validation_steps = get_number_of_steps(1605, validation_batch_size)

    from unet3d.generator import get_number_of_patches
    num_training_steps = get_number_of_steps(get_number_of_patches(data_file, training_list, patch_shape,
                                                                   patch_start_offset=training_patch_start_offset,
                                                                   patch_overlap=patch_overlap,
                                                                   is_extract_patch_agressive=is_extract_patch_agressive),
                                             batch_size)
    num_validation_steps = get_number_of_steps(get_number_of_patches(data_file, validation_list, patch_shape,
                                                                     patch_overlap=validation_patch_overlap,
                                                                     is_extract_patch_agressive=is_extract_patch_agressive),
                                               validation_batch_size)

    print("Number of training steps: ", num_training_steps)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


@threadsafe_generator
def data_generator2d(data_file, index_list, batch_size=1, n_labels=1, labels=None, patch_shape=None,
                     patch_overlap=0, patch_start_offset=None, shuffle_index_list=True,
                     skip_blank=True,
                     augment_flipud=False, augment_fliplr=False, augment_elastic=False,
                     augment_rotation=False, augment_shift=False, augment_shear=False,
                     augment_zoom=False, n_augment=False,
                     data_type_generator="combined",
                     is_extract_patch_agressive=False):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list, data_file.root.data.shape[-3:], patch_shape,
                                                 patch_overlap, patch_start_offset,
                                                 is_extract_patch_agressive=is_extract_patch_agressive)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data2d(x_list, y_list, data_file, index, patch_shape=patch_shape,
                       augment_flipud=augment_flipud, augment_fliplr=augment_fliplr,
                       augment_elastic=augment_elastic, augment_rotation=augment_rotation,
                       augment_shift=augment_shift, augment_shear=augment_shear,
                       augment_zoom=augment_zoom, data_type_generator=data_type_generator)

            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                if data_type_generator != "combined":
                    yield convert_multioutput_data2d(x_list, y_list)
                else:
                    yield convert_data2d(x_list, y_list, n_labels=n_labels, labels=labels)
                x_list = list()
                y_list = list()


def squeeze_data_from_3d_to_2d(x):
    shape = x.shape
    for i in range(len(shape)):
        if i > 1 and shape[i] == 1:
            axis = i
    return np.squeeze(x, axis=axis)


def convert_data2d(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    return squeeze_data_from_3d_to_2d(x), squeeze_data_from_3d_to_2d(y)


def convert_multioutput_data2d(x_list, y_list):
    x = squeeze_data_from_3d_to_2d(np.asarray(x_list))
    count = 0
    for y in y_list:
        if count == 0:
            y_list_whole = [squeeze_data_from_3d_to_2d(y[0])]
            y_list_core = [squeeze_data_from_3d_to_2d(y[1])]
        else:
            y_list_whole.append(squeeze_data_from_3d_to_2d(y[0]))
            y_list_core.append(squeeze_data_from_3d_to_2d(y[1]))
        count += 1

    y_whole = np.asarray(y_list_whole)
    y_core = np.asarray(y_list_core)

    return x, [y_whole, y_core]


def elastic_transform_multi2d(x, alpha, sigma, mode="constant", cval=0, is_random=False):
    """Elastic transformation for images as described in `[Simard2003] <http://deeplearning.cs.cmu.edu/pdfs/Simard.pdf>`__.
    Parameters
    -----------
    x : list of numpy.array
        List of greyscale images.
    others : args
        See ``tl.prepro.elastic_transform``.
    Returns
    -------
    numpy.array
        A list of processed images.
    """
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))

    shape = x[0].shape
    if len(shape) == 3:
        shape = (shape[0], shape[1])
    new_shape = random_state.rand(*shape)

    results = []
    for data in x:
        is_3d = False
        if len(data.shape) == 3 and data.shape[-1] == 1:
            data = data[:, :, 0]
            is_3d = True
        elif len(data.shape) == 3 and data.shape[-1] != 1:
            raise Exception("Only support greyscale image")

        if len(data.shape) != 2:
            raise AssertionError("input should be grey-scale image")

        dx = gaussian_filter((new_shape * 2 - 1), sigma,
                             mode=mode, cval=cval) * alpha
        dy = gaussian_filter((new_shape * 2 - 1), sigma,
                             mode=mode, cval=cval) * alpha

        x_, y_ = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
        # tl.logging.info(data.shape)
        if is_3d:
            results.append(map_coordinates(
                data, indices, order=1).reshape((shape[0], shape[1], 1)))
        else:
            results.append(map_coordinates(
                data, indices, order=1).reshape(shape))
    return np.asarray(results)


def augment_data2d(data, augment_flipud=False, augment_fliplr=False, augment_elastic=False,
                   augment_rotation=False, augment_shift=False, augment_shear=False, augment_zoom=False):
    """ data augumentation """
    shape_data = data[0].shape

    if np.argmin(shape_data) == 0:
        data = image_utils.move_axis_data(data, source=0, destination=-1)
    elif np.argmin(shape_data) == 1:
        data = image_utils.move_axis_data(data, source=-1, destination=0)

    if augment_flipud:
        data = tl.prepro.flip_axis_multi(
            data, axis=0, is_random=True)  # up down
    if augment_fliplr:
        data = tl.prepro.flip_axis_multi(
            data, axis=1, is_random=True)  # left right
    if augment_elastic:
        data = elastic_transform_multi2d(
            data, alpha=720, sigma=10, is_random=True)
    if augment_rotation:
        data = tl.prepro.rotation_multi(
            data, rg=20, is_random=True, fill_mode='constant')  # nearest, constant
    if augment_shift:
        data = tl.prepro.shift_multi(
            data, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    if augment_shear:
        data = tl.prepro.shear_multi(
            data, 0.05, is_random=True, fill_mode='constant')
    if augment_zoom:
        data = tl.prepro.zoom_multi(
            data, zoom_range=[0.9, 1.1], is_random=True, fill_mode='constant')

    if np.argmin(shape_data) == 0:
        data = image_utils.move_axis_data(data, source=-1, destination=0)
    elif np.argmin(shape_data) == 1:
        data = image_utils.move_axis_data(data, source=0, destination=-1)
    return data


def add_data2d(x_list, y_list, data_file, index, patch_shape=None,
               augment_flipud=False, augment_fliplr=False, augment_elastic=False,
               augment_rotation=False, augment_shift=False, augment_shear=False,
               augment_zoom=False, skip_blank=True, data_type_generator="combined"):
    """
    Adds data from the data file to the given lists of feature and target data
    :return:
    """
    data, truth = get_data_from_file(
        data_file, index, patch_shape=patch_shape)

    augment = augment_flipud or augment_fliplr or augment_elastic or augment_rotation or augment_shift or augment_shear or augment_zoom
    if augment:
        data_list = list()
        for i in range(data.shape[0]):
            data_list.append(data[i, :, :, :])
        data_list.append(truth[:, :, :])
        data_list = augment_data2d(data=data_list, augment_flipud=augment_flipud, augment_fliplr=augment_fliplr,
                                   augment_elastic=augment_elastic, augment_rotation=augment_rotation,
                                   augment_shift=augment_shift, augment_shear=augment_shear,
                                   augment_zoom=augment_zoom)
        for i in range(data.shape[0]):
            data[i, :, :, :] = data_list[i]
        truth[:, :, :] = data_list[-1]
    truth = truth[np.newaxis]

    # change here to feed more samples
    # is_added = False
    # if np.any(truth != 0):
    #     is_added = True

    is_added = True
    # change here to feed more samples

    if is_added:
        x_list.append(data)
        if data_type_generator == "cascaded":
            truth_whole, truth_core = np.copy(truth), np.copy(truth)

            truth_whole[truth_whole > 0] = 1
            truth_core[truth_core == 1] = 0
            truth_core[truth_core > 0] = 1

            y_list.append([truth_whole, truth_core])
        elif data_type_generator == "separated":
            truth_1, truth_2 = np.copy(truth), np.copy(truth)

            truth_1[truth_1 == 2] = 0

            truth_2[truth_2 == 1] = 0
            truth_2[truth_2 == 2] = 1

            y_list.append([truth_1, truth_2])

        else:
            y_list.append(truth)


def get_number_of_patches2d(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                            skip_blank=True, data_type_generator=False):

    if patch_shape:
        index_list = create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_overlap,
                                             patch_start_offset)
        count = 0
        for i, index in enumerate(index_list, 0):
            if i % 100 == 0:
                print(">> processing {}/{}, added {}/{}".format(i,
                                                                len(index_list), count, len(index_list)))
            x_list = list()
            y_list = list()
            add_data2d(x_list, y_list, data_file, index,
                       #    augment_fliplr=True,
                       #    augment_rotation=True,
                       skip_blank=skip_blank, patch_shape=patch_shape,
                       data_type_generator=data_type_generator)

            if len(x_list) > 0:
                count += 1
                # a,b = convert_data2d(x_list, y_list, n_labels=2)
                # if b[0][1].max()>0 and count % 10 == 0:
                #     from unet3d.utils.image_utils import display_array_as_image
                #     display_array_as_image(np.squeeze(a))
                #     display_array_as_image(np.squeeze(b)[0])
                #     display_array_as_image(np.squeeze(b)[1])
                #     t=0

        print(">> processing {}/{}, added {}/{}".format(i,
                                                        len(index_list), count, len(index_list)))

        return count
    else:
        return len(index_list)
