import unittest
from unittest import TestCase
import os
import glob
import shutil
import sys
import nibabel as nib
import numpy as np
import nibabel as nib


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

parent_dir = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir)))
test_dir = os.path.join(parent_dir, "brats/data_train/original")


def test_size():
    for subject_dir in glob.glob(os.path.join(test_dir, "*", "*", "*")):
        print(subject_dir)
        img = nib.load(subject_dir)
        print(img.shape)
        # data_shape = tuple([0, n_channels] + list(image_shape))
        # truth_shape = tuple([0, 1] + list(image_shape))


def get_number_volume():
    return 2

class TestImageShape(TestCase):
    def setUp(self):
        self.dir = test_dir
    def test_size(self):
        for data_folder in config["data_folders"]:
            for denoised_folder in config["denoised_folders"]:
                parent_dir = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir)))
                brats_dir = os.path.join(parent_dir, "brats")
                subject_dirs = glob.glob(os.path.join(brats_dir, data_folder, denoised_folder, "*", "*", "*.nii.gz"))
                for subject_dir in subject_dirs:
                    # print(subject_dir)
                    img = nib.load(subject_dir)
                    # print(img.shape)
                    self.assertEqual(img.shape, SHAPE)
    # def num_folder(self):




if __name__ == '__main__':
    # test_size()
    unittest.main()
