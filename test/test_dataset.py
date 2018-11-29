import unittest
from unittest import TestCase
import os
import glob
import shutil
import sys
import nibabel as nib
import numpy as np
import nibabel as nib

SHAPE = (240,240,155)

parent_dir = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir)))
test_dir = os.path.join(parent_dir, "brats/data_train/test")

def test_size():
    for subject_dir in glob.glob(os.path.join(test_dir, "*", "*", "*")):
        print(subject_dir)
        img = nib.load(subject_dir)
        print(img.shape)
        # data_shape = tuple([0, n_channels] + list(image_shape))
        # truth_shape = tuple([0, 1] + list(image_shape))
        

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.dir = test_dir

    def test_size(self):
        for subject_dir in glob.glob(os.path.join(self.dir, "*", "*", "*")):
            # print(subject_dir)
            img = nib.load(subject_dir)
            # print(img.shape)
            self.assertEqual(img.shape, SHAPE)




if __name__ == '__main__':
    # test_size()
    unittest.main()