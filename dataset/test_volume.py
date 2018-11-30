import os
import glob
import shutil
import sys
import nibabel as nib

from unet3d.utils.volume import get_bounding_box
from unet3d.utils.volume import get_shape


test_vol = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t1.nii.gz"

if __name__ == '__main__':
    print(test_vol)
    volume = nib.load(test_vol)
    volume = volume.get_fdata()
    # volume = volume.
    rmin, rmax, cmin, cmax, zmin, zmax = get_bounding_box(volume)

    print(rmin, rmax, cmin, cmax, zmin, zmax)