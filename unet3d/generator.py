import os
import copy
from random import shuffle
import itertools

import numpy as np
import time

from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data

import tensorlayer as tl
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def get_training_and_validation_and_testing_generators(data_file, batch_size, n_labels, training_keys_file,
                                                       validation_keys_file, testing_keys_file,
                                                       data_split=0.8, overwrite=False, labels=None, patch_shape=None,
                                                       validation_patch_overlap=0, training_patch_start_offset=None,
                                                       validation_batch_size=None, is_create_patch_index_list_original=True,
                                                       augment_flipud=False, augment_fliplr=False, augment_elastic=False,
                                                       augment_rotation=False, augment_shift=False, augment_shear=False,
                                                       augment_zoom=False, n_augment=0, skip_blank=False,):
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
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """

    if not validation_batch_size:
        validation_batch_size = batch_size

    training_list, validation_list, _ = get_train_valid_test_split(
        data_file, training_file=training_keys_file,
        validation_file=validation_keys_file,
        testing_file=testing_keys_file,
        data_split=0.8, overwrite=False)

    print("training_list:", training_list)

    print(">> training data generator")
    training_generator = data_generator(data_file, training_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        patch_shape=patch_shape,
                                        patch_overlap=0,
                                        patch_start_offset=training_patch_start_offset,
                                        is_create_patch_index_list_original=is_create_patch_index_list_original,
                                        augment_flipud=augment_flipud,
                                        augment_fliplr=augment_fliplr,
                                        augment_elastic=augment_elastic,
                                        augment_rotation=augment_rotation,
                                        augment_shift=augment_shift,
                                        augment_shear=augment_shear,
                                        augment_zoom=augment_zoom,
                                        n_augment=n_augment,
                                        skip_blank=skip_blank)
    print(">> valid data generator")
    validation_generator = data_generator(data_file, validation_list,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          patch_shape=patch_shape,
                                          patch_overlap=validation_patch_overlap,
                                          is_create_patch_index_list_original=is_create_patch_index_list_original,
                                          skip_blank=skip_blank
                                          )

    # Set the number of training and testing samples per epoch correctly
    # if overwrite or not os.path.exists(n_steps_file):
    print(">> compute number of training and validation steps")
    num_training_steps = get_number_of_steps(get_number_of_patches(data_file, training_list, patch_shape,
                                                                   patch_start_offset=training_patch_start_offset,
                                                                   patch_overlap=0),
                                             batch_size)
    num_validation_steps = get_number_of_steps(get_number_of_patches(data_file, validation_list, patch_shape,
                                                                     patch_overlap=validation_patch_overlap),
                                               validation_batch_size)

    print("Number of training steps: ", num_training_steps)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_train_valid_test_split(data_file, training_file, validation_file,
                               testing_file, data_split=0.8, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, testing_list = split_list(
            sample_list, split=data_split)

        training_list, validation_list = split_list(
            training_list, split=data_split)

        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        pickle_dump(testing_list, testing_file)

        return training_list, validation_list, testing_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file), pickle_load(testing_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(data_file, index_list, batch_size=1, n_labels=1, labels=None, patch_shape=None,
                   patch_overlap=0, patch_start_offset=None, shuffle_index_list=True,
                   skip_blank=True, is_create_patch_index_list_original=True,
                   augment_flipud=False, augment_fliplr=False, augment_elastic=False,
                   augment_rotation=False, augment_shift=False, augment_shear=False,
                   augment_zoom=False, n_augment=False):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list, data_file.root.data.shape[-3:], patch_shape,
                                                 patch_overlap, patch_start_offset)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, patch_shape=patch_shape,
                     augment_flipud=augment_flipud, augment_fliplr=augment_fliplr,
                     augment_elastic=augment_elastic, augment_rotation=augment_rotation,
                     augment_shift=augment_shift, augment_shear=augment_shear,
                     augment_zoom=augment_zoom)

            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels)
                x_list = list()
                y_list = list()


def get_number_of_patches(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None):
    if patch_shape:
        index_list = create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_overlap,
                                             patch_start_offset)

        return len(index_list)
    else:
        return len(index_list)


def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(
                get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape,
                                            overlap=patch_overlap, start=random_start_offset,
                                            is_extract_patch_agressive=False)
        else:
            patches = compute_patch_indices(image_shape, patch_shape,
                                            overlap=patch_overlap,
                                            is_extract_patch_agressive=False)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y = data_file.root.data[index], data_file.root.truth[index, 0]
    return x, y


def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    return x, y


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y


def elastic_transform_multi(x, alpha, sigma, mode="constant", cval=0, is_random=False):
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
    if len(shape) == 4:
        shape = (shape[0], shape[1], shape[2])
    new_shape = random_state.rand(*shape)

    results = []
    for data in x:
        is_4d = False
        if len(data.shape) == 4 and data.shape[-1] == 1:
            data = data[:, :, :, 0]
            is_4d = True
        elif len(data.shape) == 4 and data.shape[-1] != 1:
            raise Exception("Only support greyscale image")

        if len(data.shape) != 3:
            raise AssertionError("input should be grey-scale image")

        dx = gaussian_filter((new_shape * 2 - 1), sigma,
                             mode=mode, cval=cval) * alpha
        dy = gaussian_filter((new_shape * 2 - 1), sigma,
                             mode=mode, cval=cval) * alpha
        dz = np.zeros_like(dx)

        x_, y_, z_ = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))

        indices = np.reshape(y_+dy, (-1, 1)), np.reshape(x_+dx,
                                                         (-1, 1)), np.reshape(z_+dz, (-1, 1))

        # tl.logging.info(data.shape)
        if is_4d:
            results.append(map_coordinates(
                data, indices, order=1).reshape((shape[0], shape[1], 1)))
        else:
            results.append(map_coordinates(
                data, indices, order=1).reshape(shape))
    return np.asarray(results)


def augment_data(data, augment_flipud=False, augment_fliplr=False, augment_elastic=False,
                 augment_rotation=False, augment_shift=False, augment_shear=False, augment_zoom=False):
    """ data augumentation """
    if augment_flipud:
        data = tl.prepro.flip_axis_multi(
            data, axis=0, is_random=True)  # up down
    if augment_fliplr:
        data = tl.prepro.flip_axis_multi(
            data, axis=1, is_random=True)  # left right
    if augment_elastic:
        data = elastic_transform_multi(
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
    return data


def add_data(x_list, y_list, data_file, index, patch_shape=None,
             augment_flipud=False, augment_fliplr=False, augment_elastic=False,
             augment_rotation=False, augment_shift=False, augment_shear=False,
             augment_zoom=False, skip_blank=True, model_dim=3):
    """
    Adds data from the data file to the given lists of feature and target data
    :return:
    """
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)

    augment = augment_flipud or augment_fliplr or augment_elastic or augment_rotation or augment_shift or augment_shear or augment_zoom
    if augment:
        data_list = list()
        for i in range(data.shape[0]):
            data_list.append(data[i, :, :, :])
        data_list.append(truth[:, :, :])
        data_list = augment_data(data=data_list, augment_flipud=augment_flipud, augment_fliplr=augment_fliplr,
                                 augment_elastic=augment_elastic, augment_rotation=augment_rotation,
                                 augment_shift=augment_shift, augment_shear=augment_shear,
                                 augment_zoom=augment_zoom)
        for i in range(data.shape[0]):
            data[i, :, :, :] = data_list[i]
        truth[:, :, :] = data_list[-1]
    truth = truth[np.newaxis]
    is_added = False
    if model_dim == 3:
        is_added = True
    if model_dim == 25:
        truth_slice = truth[..., int((patch_shape[-1]-1)/2)]
        # if np.any(truth_slice != 0):
        if np.any(data != 0):
            is_added = True
    # if model_dim == 2 and np.any(truth != 0):
    # if model_dim==2 and np.any(data != 0):
    if model_dim==2:
        is_added = True
    if is_added:
        x_list.append(data)
        y_list.append(truth)
