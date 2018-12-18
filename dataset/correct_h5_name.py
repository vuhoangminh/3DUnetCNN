import os
import glob
from unet3d.utils.path_utils import get_project_dir, get_h5_training_dir
from brats.config import config

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


for h5_path in glob.glob(os.path.join(BRATS_DIR, "database", "*", "*.h5")):
    # print(h5_path)
    print("old name:", h5_path)
    if "norm-minh" in h5_path:
        new_name = h5_path.replace("norm-minh", "norm-01_hist-1")
    if "norm-z" in h5_path:         
        new_name = h5_path.replace("norm-z", "norm-z-old_hist-0")
    print(">> rename to:", new_name)        
    os.rename(h5_path, new_name)
