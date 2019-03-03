import os
import glob
import shutil
from random import shuffle
import numpy as np
np.random.seed(1988)

import unet3d.utils.print_utils as print_utils
import unet3d.utils.path_utils as path_utils

from projects.drive.config import config

import unet3d.utils.args_utils as get_args

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(
    CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])
RAW_DIR = os.path.join(PROJECT_DIR, "raw/DRIVE")


def clean_name(name):
    case = name.split('_')
    return case[0]


def get_manual_mask_name(name, dataset="train"):
    seg1 = "{}_manual1.gif".format(name)
    seg2 = "{}_manual2.gif".format(name)
    if dataset == "train":
        mask = "{}_training_mask.gif".format(name)
    else:
        mask = "{}_test_mask.gif".format(name)
    return seg1, seg2, mask


def move_organize_training(dir_in, dir_out, dataset="train"):
    train_dir = os.path.join(dir_out, "data_train/original")
    valid_dir = os.path.join(dir_out, "data_valid/original")
    test_dir = os.path.join(dir_out, "data_test/original")

    if dataset == "train":
        images_dir = dir_in + "/training/images/"
        seg1_dir = dir_in + "/training/1st_manual/"
        seg2_dir = dir_in + "/training/2nd_manual/"
        mask_dir = dir_in + "/training/mask/"
    else:
        images_dir = dir_in + "/test/images/"
        seg1_dir = dir_in + "/test/1st_manual/"
        seg2_dir = dir_in + "/test/2nd_manual/"
        mask_dir = dir_in + "/test/mask/"
    

    img_dirs = glob.glob(os.path.join(images_dir, "*"))
    
    if dataset == "train":
        shuffle(img_dirs)

    num_train = 14
    count = 0

    for img_dir in img_dirs:
        print(">> processing", img_dir)
        img_full_name = path_utils.get_filename(img_dir)
        img_name = clean_name(img_full_name)

        seg1_name, seg2_name, mask_name = get_manual_mask_name(img_name, dataset=dataset)
    
        seg1_path = seg1_dir + seg1_name
        seg2_path = seg2_dir + seg2_name
        mask_path = mask_dir + mask_name

        if dataset == "train" and count<num_train:
            img_path_dst = os.path.join(train_dir, img_name, img_full_name)
            seg1_path_dst = os.path.join(train_dir, img_name, seg1_name)
            mask_path_dst = os.path.join(train_dir, img_name, mask_name)
            print("copy to", img_path_dst)
            print("copy to", seg1_path_dst)
            print("copy to", mask_path_dst)
            if not os.path.exists(os.path.join(train_dir, img_name)):
                os.makedirs(os.path.join(train_dir, img_name))
            shutil.copy2(img_dir, img_path_dst)
            shutil.copy2(seg1_path, seg1_path_dst)
            shutil.copy2(mask_path, mask_path_dst)
            count = count + 1
        elif dataset == "train" and count>=num_train:
            img_path_dst = os.path.join(valid_dir, img_name, img_full_name)
            seg1_path_dst = os.path.join(valid_dir, img_name, seg1_name)
            mask_path_dst = os.path.join(valid_dir, img_name, mask_name)
            print("copy to", img_path_dst)
            print("copy to", seg1_path_dst)
            print("copy to", mask_path_dst)
            if not os.path.exists(os.path.join(valid_dir, img_name)):
                os.makedirs(os.path.join(valid_dir, img_name))
            shutil.copy2(img_dir, img_path_dst)
            shutil.copy2(seg1_path, seg1_path_dst)
            shutil.copy2(mask_path, mask_path_dst)
            count = count + 1
        else:
            img_path_dst = os.path.join(test_dir, img_name, img_full_name)
            seg1_path_dst = os.path.join(test_dir, img_name, seg1_name)
            seg2_path_dst = os.path.join(test_dir, img_name, seg2_name)
            mask_path_dst = os.path.join(test_dir, img_name, mask_name)
            print("copy to", img_path_dst)
            print("copy to", seg1_path_dst)
            print("copy to", seg2_path_dst)
            print("copy to", mask_path_dst)
            if not os.path.exists(os.path.join(test_dir, img_name)):
                os.makedirs(os.path.join(test_dir, img_name))
            shutil.copy2(img_dir, img_path_dst)
            shutil.copy2(seg1_path, seg1_path_dst)
            shutil.copy2(seg2_path, seg2_path_dst)
            shutil.copy2(mask_path, mask_path_dst)
            count = count + 1


def main():
    dir_in = RAW_DIR
    dir_out = os.path.join(PROJECT_DIR, "projects/drive/database")

    # move_organize_training(dir_in, dir_out, dataset="train")
    move_organize_training(dir_in, dir_out, dataset="test")


if __name__ == "__main__":
    main()
