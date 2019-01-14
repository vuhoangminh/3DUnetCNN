import os
import nibabel as nib
import numpy as np
import time
import gc
from unet3d.training import load_old_model
from unet3d.utils.model_utils import load_model_multi_gpu

from unet3d.utils.path_utils import get_project_dir
from brats.evaluate import get_whole_tumor_mask, get_enhancing_tumor_mask, get_tumor_core_mask
from brats.config import config, config_unet, config_dict

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])

model_path = BRATS_DIR + "/database/model/finetune/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-z_hist-0_ps-160-192-128_isensee3d_crf-0_loss-minh_model.h5"


# t0 = time.time()
# model = load_old_model(model_path)
# gc.collect()
# t1 = time.time()

# print("time load old:", t1-t0)

# t0 = time.time()
# model = load_model_multi_gpu(model_path)
# gc.collect()
# t1 = time.time()

# print("time load new:", t1-t0)


path = BRATS_DIR + "/database/prediction/csv/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_ps-128-128-128_isensee3d_crf-0_loss-minh_model.csv"
import pandas as pd

df = pd.read_csv(path)

header = ("dice_WholeTumor", "dice_TumorCore", "dice_EnhancingTumor")

df1 = df.dice_WholeTumor.T._values
df2 = df.dice_TumorCore.T._values
df3 = df.dice_EnhancingTumor.T._values

# df1[:,:-1] = df2
# df1[:,:-1] = df3

scores = np.zeros((df1.size,3))

scores[:,0] = df1
scores[:,1] = df2
scores[:,2] = df3


print(scores)

# print(model_path)


# depth = 4
# for layer_depth in reversed(range(depth)):
#     kernel_size = 3 + layer_depth*2
#     filters = 8*2**(depth-layer_depth-1)
#     print(kernel_size, filters)
