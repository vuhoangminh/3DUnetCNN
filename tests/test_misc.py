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

path = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/database/prediction/csv/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_ps-128-128-128_isensee3d_crf-0_loss-minh_model.csv"
import pandas as pd

df = pd.read_csv(path)

scores = dict()
for index, score in enumerate(df.columns):
    if "dice" in score:
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]

print(scores)

print(model_path)