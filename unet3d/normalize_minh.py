# -*- coding: utf-8 -*-
"""
Created on Mon Dec 03
Copyright (c) 2018, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

import numpy as np
from skimage import data, img_as_float
from skimage import exposure
import numpy as np
import cv2


def normalize_mean_std(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    volume_norm = (volume-mean)/std
    return volume_norm


def normalize_to_0_1 (volume):
    volume_temp = volume
    volume_temp[volume_temp<0] = 0

    x_min = np.min(volume_temp)
    x_max = np.max(volume_temp)

    volume_norm = (volume - x_min)/(x_max-x_min)
    return volume_norm

def perform_histogram_equalization(volume):
    return exposure.equalize_hist(volume)    

def perform_adaptive_histogram_equalization(volume):
    # volume_temp = volume.flatten()
    volume_temp = volume.reshape((volume.shape[0], volume.shape[1]*volume.shape[2]))
    volume_adap = exposure.equalize_adapthist(volume_temp,clip_limit=0.01)    
    return volume_adap.reshape((volume.shape[0],volume.shape[1],volume.shape[2]))    

    # volume_temp = volume.reshape((volume.shape[0]*volume.shape[1]*volume.shape[2], 1))
    # volume_adap = exposure.equalize_adapthist(volume_temp)    
    # return volume_adap.reshape((volume.shape[0],volume.shape[1],volume.shape[2]))    

def perform_adaptive_histogram_equalization_opencv(volume):
    # volume_temp = volume.flatten()
    volume_temp = volume.reshape((volume.shape[0], volume.shape[1]*volume.shape[2]))
    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(8,8))

    h,w = volume_temp.shape
    vis2 = cv2.CreateMat(h, w, cv2.CV_32FC3)
    vis0 = cv2.fromarray(vis)
    cv2.CvtColor(vis0, vis2, cv2.CV_GRAY2BGR)
    volume_adap = clahe.apply(volume_temp)   
    return volume_adap.reshape((volume.shape[0],volume.shape[1],volume.shape[2]))    


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)