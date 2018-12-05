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
from unet3d.normalize_minh import hist_match, hist_match_non_zeros


laptop_save_dir = "C:/Users/minhm/Desktop/temp/"
desktop_save_dir = "/home/minhvu/Desktop/temp/"
save_dir = desktop_save_dir
laptop_dir = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/"
desktop_dir = "/home/minhvu/github/3DUnetCNN_BRATS/brats/data_train/denoised_preprocessed/HGG/"
image_dir = desktop_dir


volume_path = image_dir + "Brats18_2013_2_1/t1.nii.gz"
volume_path = image_dir + "Brats18_2013_2_1/t1ce.nii.gz"
volume_path = image_dir + "Brats18_2013_2_1/flair.nii.gz"
volume_path = image_dir + "Brats18_2013_2_1/t2.nii.gz"

source_path = image_dir + "Brats18_2013_3_1/flair.nii.gz"
template_path = image_dir + "Brats18_2013_2_1/flair.nii.gz"

truth_path = image_dir + "Brats18_2013_2_1/truth.nii.gz"
mask_path = get_mask_path(volume_path)
temp_path = save_dir + "temp.nii.gz"
temp_norm_path = save_dir + "temp_norm.nii.gz"
temp_norm_mean_path = save_dir + "temp_norm_mean.nii.gz"
temp_hist_path = save_dir + "temp_hist.nii.gz"
temp_adap_path = save_dir + "temp_adap.nii.gz"
temp_adap_cv_path = save_dir + "temp_adap_cv.nii.gz"
temp_hist_match_path = save_dir + "temp_hist_match.nii.gz"


temp_source_path = save_dir + "source.nii.gz"
temp_template_path = save_dir + "template.nii.gz"
temp_source_norm_path = save_dir + "source_norm.nii.gz"
temp_template_norm_path = save_dir + "template_norm.nii.gz"
new_temp_hist_match_path = save_dir + "new_temp_hist_match.nii.gz"


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


def save_nib(volume, path):
    volume_temp = nib.Nifti1Image(volume, affine=affine)
    nib.save(volume_temp, path)    




def main():  

    # idx = np.argwhere(truth > 0)
    volume_temp = np.multiply(volume, mask)
    volume_norm = normalize_to_0_1(volume_temp)
    volume_norm_mean = normalize_mean_std(volume_temp)
    volume_hist = perform_histogram_equalization(volume_norm)
    volume_adap = perform_adaptive_histogram_equalization(volume_norm)
    volume_hist_match = hist_match(source, template)
    source_hist_match = hist_match_non_zeros(source, template)
    # volume_adap_cv = perform_adaptive_histogram_equalization_opencv(volume_norm)


    save_nib(volume_temp, temp_path)
    save_nib(volume_norm, temp_norm_path)
    save_nib(volume_norm_mean, temp_norm_mean_path)
    save_nib(volume_hist, temp_hist_path)
    save_nib(volume_adap, temp_adap_path)
    save_nib(volume_hist_match, temp_hist_match_path)
    save_nib(source, temp_source_path)
    save_nib(template, temp_template_path)
    save_nib(source_hist_match, new_temp_hist_match_path)

def test():
    
    new_temp_hist_match = hist_match_non_zeros(source, template)
    

if __name__ == '__main__':
    main()
    # test()
