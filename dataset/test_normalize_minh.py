import os
import glob
import shutil
import sys
import nibabel as nib
import numpy as np

from dataset.mask_dataset import get_mask_path
from operator import itemgetter 

from unet3d.normalize_minh import normalize_mean_std
from unet3d.normalize_minh import normalize_to_0_1
from unet3d.normalize_minh import perform_histogram_equalization
from unet3d.normalize_minh import perform_adaptive_histogram_equalization
from unet3d.normalize_minh import perform_adaptive_histogram_equalization_opencv
from unet3d.normalize_minh import hist_match

volume_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t1.nii.gz"
volume_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t1ce.nii.gz"
volume_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/flair.nii.gz"
volume_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/t2.nii.gz"

source_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/original/HGG/Brats18_2013_3_1/flair.nii.gz"
template_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/original/HGG/Brats18_2013_2_1/flair.nii.gz"

truth_path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/Brats18_2013_2_1/truth.nii.gz"
mask_path = get_mask_path(volume_path)
temp_path = "C:/Users/minhm/Desktop/temp.nii.gz"
temp_norm_path = "C:/Users/minhm/Desktop/temp_norm.nii.gz"
temp_norm_mean_path = "C:/Users/minhm/Desktop/temp_norm_mean.nii.gz"
temp_hist_path = "C:/Users/minhm/Desktop/temp_hist.nii.gz"
temp_adap_path = "C:/Users/minhm/Desktop/temp_adap.nii.gz"
temp_adap_cv_path = "C:/Users/minhm/Desktop/temp_adap_cv.nii.gz"
temp_hist_match_path = "C:/Users/minhm/Desktop/temp_hist_match.nii.gz"


volume = nib.load(volume_path)
affine = volume.affine
volume = volume.get_fdata()
truth = nib.load(truth_path)
truth = truth.get_fdata()
mask = nib.load(mask_path)
mask = mask.get_fdata()
source = nib.load(source_path)
source = source.get_fdata()
template = nib.load(template_path)
template = template.get_fdata()


def main():  

    idx = np.argwhere(truth > 0)
    volume_temp = np.multiply(volume, mask)
    volume_norm = normalize_to_0_1(volume_temp)
    volume_norm_mean = normalize_mean_std(volume_temp)
    volume_hist = perform_histogram_equalization(volume_norm)
    volume_adap = perform_adaptive_histogram_equalization(volume_norm)
    volume_hist_match = hist_match(source, template)
    # volume_adap_cv = perform_adaptive_histogram_equalization_opencv(volume_norm)

    volume_temp = nib.Nifti1Image(volume_temp, affine=affine)
    nib.save(volume_temp, temp_path)

    volume_norm = nib.Nifti1Image(volume_norm, affine=affine)
    nib.save(volume_norm, temp_norm_path)

    volume_norm_mean = nib.Nifti1Image(volume_norm_mean, affine=affine)
    nib.save(volume_norm_mean, temp_norm_mean_path)

    volume_hist = nib.Nifti1Image(volume_hist, affine=affine)
    nib.save(volume_hist, temp_hist_path)

    volume_adap = nib.Nifti1Image(volume_adap, affine=affine)
    nib.save(volume_adap, temp_adap_path)

    volume_hist_match = nib.Nifti1Image(volume_hist_match, affine=affine)
    nib.save(volume_hist_match, temp_hist_match_path)


    # volume_adap_cv = nib.Nifti1Image(volume_adap_cv, affine=affine)
    # nib.save(volume_adap_cv, temp_adap_cv_path)

if __name__ == '__main__':
    main()
