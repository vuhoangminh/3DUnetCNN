import os
import glob
import shutil
import sys
import nibabel as nib

volume_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t1.nii.gz"
truth_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/truth.nii.gz"
volume = nib.load(volume_path)
volume = volume.get_fdata()
truth = nib.load(truth_path)
truth = truth.get_fdata()


def main():
    print("hello")


if __name__ == '__main__':
    main()
