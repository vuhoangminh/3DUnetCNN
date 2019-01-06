import os
import nibabel as nib
import numpy as np
from numpy.core.umath_tests import inner1d
from medpy.metric.binary import hd, sensitivity, specificity, dc
from unet3d.utils.path_utils import get_project_dir
from brats.evaluate import get_whole_tumor_mask, get_enhancing_tumor_mask, get_tumor_core_mask
from brats.config import config, config_unet, config_dict

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])

voxelspacings = [(0.9375, 1.5, 0.9375), (1, 1.5, 1), (0.8371, 1.5, 0.8371)]

prediction_path = BRATS_DIR + "/database/prediction/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_ps-128-128-128_isensee3d_crf-0_loss-weighted_model/validation_case_0/prediction.nii.gz"
truth_path = BRATS_DIR + "/database/prediction/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_ps-128-128-128_isensee3d_crf-0_loss-weighted_model/validation_case_0/truth.nii.gz"

def HausdorffDist(A,B):
    # Hausdorf Distance: Compute the Hausdorff distance between two point
    # clouds.
    # Let A and B be subsets of metric space (Z,dZ),
    # The Hausdorff distance between A and B, denoted by dH(A,B),
    # is defined by:
    # dH(A,B) = max(h(A,B),h(B,A)),
    # where h(A,B) = max(min(d(a,b))
    # and d(a,b) is a L2 norm
    # dist_H = hausdorff(A,B)
    # A: First point sets (MxN, with M observations in N dimension)
    # B: Second point sets (MxN, with M observations in N dimension)
    # ** A and B may have different number of rows, but must have the same
    # number of columns.
    #
    # Edward DongBo Cui; Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
    return(dH)


prediction = nib.load(prediction_path)
prediction = prediction.get_data()

truth = nib.load(truth_path)
truth = truth.get_data()

a = get_whole_tumor_mask(prediction).astype(int)
b = get_whole_tumor_mask(truth).astype(int)
hd_wt = HausdorffDist(a, b)
print(hd_wt)