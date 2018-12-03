import os
import glob
import shutil
import sys
import nibabel as nib
import argparse
import pandas as pd

from brats.prepare_data import get_save_to_dir
from brats.config import config

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

current_dir = os.path.realpath(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
brats_dir = os.path.join(os.path.abspath(
    os.path.join(parent_dir, os.pardir)), "brats")

columns = ["dataset",
           "folder",
           "name",
           "modality",
           "size",
           "shape",
           "bounding_box",
           "size_bounding_box",
           "n_non_zeros_pixel",
           "n_zeros_pixel",
           "mean_non_zeros_pixel",
           "std_non_zeros_pixel",
           "number_occurrences_label",
           "n_unique_label",
           "max_intensity",
           "min_intensity",
           "n_non_zeros_background",
           "n_zeros_non_background"
           ]

volume_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t1.nii.gz"


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


def get_header_info(path):
    folders = split_dos_path_into_components(path)
    N = len(folders)
    dataset = folders[N-4]
    folder = folders[N-3]
    name = folders[N-2]
    modality = folders[N-1].replace(".nii.gz", "")
    return dataset, folder, name, modality


def get_data_dir(brats_dir, data_folder="data_train", dataset="original"):
    return os.path.join(brats_dir, data_folder, dataset)


def main(overwrite=False):
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=config["dataset"],
                        default="original",
                        help="dataset type")
    parser.add_argument('-f', '--folder', type=str,
                        choices=config["data_folders"],
                        default="data_train",
                        help="data folders")

    args = parser.parse_args()
    dataset = args.dataset
    data_folder = args.folder
    data_dir = get_data_dir(brats_dir=brats_dir,
                            data_folder=data_folder, dataset=dataset)
    subject_dirs = glob.glob(os.path.join(data_dir, "*", "*", "*.nii.gz"))

    index = range(0, len(subject_dirs)-1, 1)
    df = pd.DataFrame(index=index, columns=columns)
    # df = df.fillna(0)
    # print(df)

    for i in range(len(subject_dirs)):
        subject_dir = subject_dirs[i]
        print(subject_dir)

        dataset, folder, name, modality = get_header_info(volume_path)

        df["dataset"][i] = dataset
        df["folder"][i] = folder
        df["name"][i] = name
        df["modality"][i] = modality

    print(df)         




def test_pandas():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
    print(df)
    df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
    df = df.append(df2)
    print(df)


if __name__ == "__main__":
    main(False)
    # test_pandas()
