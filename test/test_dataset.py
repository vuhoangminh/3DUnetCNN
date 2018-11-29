import unittest
from unittest import TestCase
import os
import glob
import shutil
import sys
import nibabel as nib
import numpy as np
import ntpath

import sys
sys.path.append("C://Users//minhm//Documents//GitHub//3DUnetCNN_BRATS")

from unet3d.augment import get_bounding_box

config = dict()
config["env"] = "SERVER"  # change this to "FULL" if you want to run full
# config["mode"] = "TEST"  # change this to "FULL" if you want to run full
config["mode"] = "FULL"  # change this to "FULL" if you want to run full
config["data_folders"] = ["data_train", "data_valid"]
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
# change this if you want to only use some of the modalities
config["training_modalities"] = config["all_modalities"]
config["truth_old"] = ["seg"]
config["truth"] = ["truth"]
config["groundtruth_modalities"] = config["truth_old"] + config["truth"]
if config["mode"] == "TEST":
    config["denoised_folders"] = ["test"]
else:
    config["denoised_folders"] = ["original", "preprocessed",
                                  "denoised_original", "denoised_preprocessed"]
config["original_folder"] = ["original_bak"]


SHAPE = (240, 240, 155)

parent_dir = os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), os.pardir)))
test_dir = os.path.join(parent_dir, "brats/data_train/original")
test_vol = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t1.nii.gz"


class TestImageShape(TestCase):
    def setUp(self):
        pass

    def test_size(self):
        for data_folder in config["data_folders"]:
            for denoised_folder in config["denoised_folders"]:
                parent_dir = os.path.abspath(os.path.join(
                    os.path.join(os.path.dirname(__file__), os.pardir)))
                brats_dir = os.path.join(parent_dir, "brats")
                subject_dirs = glob.glob(os.path.join(
                    brats_dir, data_folder, denoised_folder, "*", "*", "*.nii.gz"))
                for subject_dir in subject_dirs:
                    # print(subject_dir)
                    img = nib.load(subject_dir)
                    # print(img.shape)
                    self.assertEqual(img.shape, SHAPE)

    def test_num_folder(self):
        for data_folder in config["data_folders"]:
            parent_dir = os.path.abspath(os.path.join(
                os.path.join(os.path.dirname(__file__), os.pardir)))
            brats_dir = os.path.join(parent_dir, "brats")
            original_dir = glob.glob(os.path.join(
                brats_dir, data_folder, config["original_folder"][0], "*", "*", "*.nii.gz"))
            num_volume_original = len(original_dir)
            for denoised_folder in config["denoised_folders"]:
                subject_dirs = glob.glob(os.path.join(
                    brats_dir, data_folder, denoised_folder, "*", "*", "*.nii.gz"))
                num_volume = len(subject_dirs)
                self.assertEqual(num_volume, num_volume_original)

    def test_folder_name(self):
        for data_folder in config["data_folders"]:
            modalities_names = list()
            for modalities in config["all_modalities"]:
                modalities_names.append(modalities + ".nii.gz")
            if "train" in data_folder:
                modalities_names.append("truth.nii.gz")
            parent_dir = os.path.abspath(os.path.join(
                os.path.join(os.path.dirname(__file__), os.pardir)))
            brats_dir = os.path.join(parent_dir, "brats")
            for denoised_folder in config["denoised_folders"]:
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
    # unittest.main()
    test_bounding_box()