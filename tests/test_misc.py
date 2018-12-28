from unet3d.utils.patches import compute_patch_indices
import itertools
from itertools import permutations, repeat
import nibabel as nib
import unet3d.utils.path_utils as path_utils

from unet3d.data import write_data_to_file, open_data_file
# a = [1, 2, 3]
# b = [4, 5, 6]


# c = list(itertools.product(a, b))
# print(list(itertools.product(a, b)))

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(
        shape[1]), np.arange(shape[2]))
    print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx,
                                                    (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


# def test_elastic():
#     in_path = "C:/Users/minhm/Desktop/temp/template.nii.gz"

#     temp_in = nib.load(in_path)

#     temp_out = elastic_transform


#     source_hist_match = nib.Nifti1Image(source_hist_match, affine=affine)
#     nib.save(source_hist_match, normalize_minh_file_path)


def test_path_utils():
    datatype = "data"
    challenge = "brats"
    year = "2018"
    input_shape = "160-192-128"
    is_bias_correction = 1
    is_denoise = 0
    is_normalize = "z"
    crop = 1
    path = path_utils.get_training_h5_filename(datatype, challenge=challenge,
                                               year=year, input_shape=input_shape,
                                               crop=crop, is_bias_correction=is_bias_correction,
                                               is_denoise=is_denoise, is_normalize=is_normalize)
    print(path)


# test_path_utils()


def test_read_h5():
    h5_path = "/home/minhvu/github/3DUnetCNN_BRATS/brats/database/data/brats2018_denoised_original_normalize_mean_std/brats_data.h5"
    data_file_opened = open_data_file(h5_path)

    print(data_file_opened)


shape = (128, 128, 128)


def get_shape_string(input_shape):
    shape_string = ""
    for i in range(len(input_shape)-1):
        shape_string = shape_string + str(input_shape[i]) + "-"
    shape_string = shape_string + str(input_shape[-1])

    return shape_string


indices = compute_patch_indices((160, 192, 128), (128, 128, 128),
                                overlap=0, start=None, is_extract_patch_agressive=False)

print(indices) 

i_list = indices.tolist()
print(i_list)

new_list = [[-48,32,0], [80,32,0]]

i_list.extend(new_list)

print(i_list)


final_list = np.asarray(i_list)
print(final_list)


a = (128,128,128)
print(type(a))



indices = compute_patch_indices((160, 192, 128), (128, 128, 128),
                                overlap=0, start=None, 
                                is_extract_patch_agressive=False,
                                is_predict=True)

print(indices)                                
