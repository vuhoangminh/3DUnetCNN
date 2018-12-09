import unittest
from unittest import TestCase
import os
import glob
import shutil
import sys
import nibabel as nib
import numpy as np
import ntpath

from unet3d.utils.path_utils import get_project_dir, get_h5_image_path
from unet3d.utils.utils import str2bool

from unet3d.utils.volume import get_bounding_box
from brats.config import config

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])

SHAPE = (240, 240, 155)

test_dir = os.path.join(PROJECT_DIR, "brats/data_train/original")
test_vol = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t1.nii.gz"


class TestImageShape(TestCase):
    def setUp(self):
        pass

    def test_size(self):
        for data_folder in config["data_folders"]:
            for denoised_folder in config["dataset"]+config["dataset_minh_normalize"]:
                PROJECT_DIR = os.path.abspath(os.path.join(
                    os.path.join(os.path.dirname(__file__), os.pardir)))
                brats_dir = os.path.join(PROJECT_DIR, "brats")
                subject_dirs = glob.glob(os.path.join(
                    brats_dir, data_folder, denoised_folder, "*", "*", "*.nii.gz"))
                for subject_dir in subject_dirs:
                    # print(subject_dir)
                    img = nib.load(subject_dir)
                    # print(img.shape)
                    self.assertEqual(img.shape, SHAPE)

    def test_num_folder(self):
        for data_folder in config["data_folders"]:
            PROJECT_DIR = os.path.abspath(os.path.join(
                os.path.join(os.path.dirname(__file__), os.pardir)))
            brats_dir = os.path.join(PROJECT_DIR, "brats")
            original_dir = glob.glob(os.path.join(
                brats_dir, data_folder, "original", "*", "*", "*.nii.gz"))
            num_volume_original = len(original_dir)
            for denoised_folder in config["dataset"]+config["dataset_minh_normalize"]:
                if "test" not in denoised_folder:
                    subject_dirs = glob.glob(os.path.join(
                        brats_dir, data_folder, denoised_folder, "*", "*", "*.nii.gz"))
                    num_volume = len(subject_dirs)
                    if num_volume != num_volume_original:
                        print(denoised_folder)
                    self.assertEqual(num_volume, num_volume_original)

    def test_folder_name(self):
        for data_folder in config["data_folders"]:
            modalities_names = list()
            for modalities in config["all_modalities"] + config["mask"]:
                modalities_names.append(modalities + ".nii.gz")
            if "train" in data_folder:
                modalities_names.append("truth.nii.gz")
            PROJECT_DIR = os.path.abspath(os.path.join(
                os.path.join(os.path.dirname(__file__), os.pardir)))
            brats_dir = os.path.join(PROJECT_DIR, "brats")
            for denoised_folder in config["dataset"]+config["dataset_minh_normalize"]:
                subject_dirs = glob.glob(os.path.join(
                    brats_dir, data_folder, denoised_folder, "*", "*"))
                new_words = list()
                for subject_dir in subject_dirs:
                    file_dirs = glob.glob(
                        os.path.join(subject_dir, "*.nii.gz"))
                    for file_dir in file_dirs:
                        volume_name = ntpath.basename(file_dir)
                        new_words.append(volume_name)
                        self.assertIn(volume_name, modalities_names)
                    number_file = len(set(new_words))
                    number_modalities = len(modalities_names)
                    if number_file != number_modalities:
                        print(subject_dir)
                    self.assertEqual(number_file, number_modalities)


def test_bounding_box():
    print(test_vol)
    volume = nib.load(test_vol)
    volume = volume.get_fdata()
    # volume = volume.
    rmin, rmax, cmin, cmax, zmin, zmax = get_bounding_box(volume)

    print(rmin, rmax, cmin, cmax, zmin, zmax)


if __name__ == '__main__':
    # test_size()
    unittest.main()
    # test_bounding_box()
