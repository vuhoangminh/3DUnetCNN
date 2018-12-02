import os
import glob
import shutil
import sys
import nibabel as nib

from unet3d.utils.volume import get_bounding_box
from unet3d.utils.volume import get_size_bounding_box
from unet3d.utils.volume import get_shape
from unet3d.utils.volume import get_non_zeros_pixel, get_zeros_pixel
from unet3d.utils.volume import compute_mean_non_zeros_pixel, compute_std_non_zeros_pixel
from unet3d.utils.volume import count_number_occurrences_label
from unet3d.utils.volume import get_unique_label
from unet3d.utils.volume import get_max_min_intensity
from unet3d.utils.volume import count_non_zeros_background, count_zeros_non_background
from unet3d.utils.volume import get_size
from unet3d.utils.volume import get_truth_path, get_volume_paths
from unet3d.utils.print_utils import print_processing, print_section, print_separator


volume_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t1.nii.gz"
truth_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/truth.nii.gz"
volume = nib.load(volume_path)
volume = volume.get_fdata()
truth = nib.load(truth_path)
truth = truth.get_fdata()


def test_one_volume():
    # Test image
    print_separator()
    print_processing("volume")
    bounding_box_volume = get_bounding_box(volume)
    size_bounding_box_volume = get_size_bounding_box(volume)
    shape = get_shape(volume)
    n_nonzeros = get_non_zeros_pixel(volume)
    n_zeros = get_zeros_pixel(volume)
    mean = compute_mean_non_zeros_pixel(volume)
    std = compute_std_non_zeros_pixel(volume)
    max_whole, min_zeros, min_nonzeros = get_max_min_intensity(volume)
    size_image = get_size(volume_path)

    print("coordinates of bounding box of volume:", bounding_box_volume)
    print("size of bounding box of volume:", size_bounding_box_volume)
    print("shape of image:", shape)
    print("number of non-zeros elements:", n_nonzeros)
    print("number of zeros elements:", n_zeros)
    print("mean of non-zeros elements:", mean)
    print("std of non-zeros elements:", std)
    print("max, min-zeros, min-nonzeros:", max_whole, min_zeros, min_nonzeros)
    print("size of file: {}mb".format(size_image))

    # Test truth
    print_separator()
    print_processing("truth")
    bin_count = count_number_occurrences_label(truth)
    n_unique_label = get_unique_label(truth)
    bounding_box_truth = get_bounding_box(truth)
    size_bounding_box_truth = get_size_bounding_box(truth)

    print("coordinates of bounding box of truth:", bounding_box_truth)
    print("size of bounding box of truth:", size_bounding_box_truth)
    print("number of occurrences of each label:", bin_count)
    print("number of unique label:", n_unique_label)

    # Test image and truth
    print_separator()
    print_processing("volume and truth")
    n_non_zeros_background = count_non_zeros_background(volume, truth)
    n_zeros_non_background = count_zeros_non_background(volume, truth)

    print("number of non zeros but label as background:", n_non_zeros_background)
    print("number of zeros but label as non background:", n_zeros_non_background)


def test():
    print(volume_path)
    truth_path_temp = get_truth_path(volume_path)
    print(truth_path_temp)
    volume_paths = get_volume_paths(truth_path)
    for path in volume_paths:
        print(path)


if __name__ == '__main__':
    # main()
    test_one_volume()
#     test()
