import unet3d.utils.args_utils as get_args
from projects.drive.config import config
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


def rename(path_in):
    path_out = path_utils.get_parent_dir(path_in)
    path_out = path_out.replace("data_train_original", "data_train")
    path_utils.make_dir(path_out)
    name = path_utils.get_filename(path_in)
    
    if "mask" in name:
        name_out = "mask.gif"
    elif "manual1" in name:
        name_out = "truth.gif"
    else:
        name_out = "t1.tif"
    
    if "manual2" not in name:
        path_out = "{}/{}".format(path_out, name_out)
        print("copy {} to {}".format(name, name_out))
        shutil.copy(path_in, path_out)


def main():
    img_dirs = glob.glob(os.path.join(DATASET_DIR, "data_train_original/*/*/*"))
    for img_path in img_dirs:
        rename(img_path)


if __name__ == "__main__":
    main()
