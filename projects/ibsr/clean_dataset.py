import unet3d.utils.args_utils as get_args
from projects.ibsr.config import config
import unet3d.utils.path_utils as path_utils
import unet3d.utils.print_utils as print_utils
import os
import glob
import shutil
from random import shuffle
import numpy as np
np.random.seed(1988)


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(
    CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])
RAW_DIR = os.path.join(PROJECT_DIR, "raw/IBSR18")


def rename(path_in):
    name = path_utils.get_filename(path_in)
    parent_dir = path_utils.get_parent_dir(path_in)
    if "ana_strip" in name:
        name_out = "t1.nii.gz"
    elif "ana_brainmask" in name:
        name_out = "mask.nii.gz"
    elif "segTRI_ana" in name:
        name_out = "truth.nii.gz"

    path_out = "{}/{}".format(parent_dir, name_out)

    print("rename {} to {}".format(name, name_out))
    shutil.move(path_in, path_out)


def main():
    img_dirs = glob.glob(os.path.join(DATASET_DIR, "*/*/*/*.nii.gz"))
    for img_path in img_dirs:
        rename(img_path)


if __name__ == "__main__":
    main()
