import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random
import itertools


def get_shape(image):
    return image.shape


def get_bounding_box(volume):
    r = np.any(volume, axis=(1, 2))
    c = np.any(volume, axis=(0, 2))
    z = np.any(volume, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin.T, rmax.T, cmin.T, cmax.T, zmin.T, zmax.T
