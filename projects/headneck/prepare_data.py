import os
import glob
import argparse

from unet3d.data import write_data_to_file
from unet3d.utils.print_utils import print_separator, print_section

from projects.headneck.config import config, config_dict
from unet3d.utils.path_utils import get_project_dir, get_h5_training_dir
from unet3d.utils.path_utils import get_training_h5_filename, get_shape_string, get_shape_from_string
from unet3d.utils.utils import str2bool

import unet3d.utils.args_utils as get_args

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def get_dataset(is_test="1", is_denoise="0"):
    if str2bool(is_test):
        return "test"
    else:
        if is_denoise == "bm4d":
            return "denoised"
        else:
            return "original"


def fetch_training_data_files(dataset):
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "database/data_train", dataset, "*")):
        subject_files = list()
        for modality in config["training_modalities"] + config["truth"]:
            subject_files.append(os.path.join(
                subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files


def prepare_data(args):

    data_dir = get_h5_training_dir(BRATS_DIR, "data")

    # make dir
    if not os.path.exists(data_dir):
        print_separator()
        print("making dir", data_dir)
        os.makedirs(data_dir)

    print_section("convert input images into an hdf5 file")

    data_filename = get_training_h5_filename("data", args)

    print(data_filename)

    data_file_path = os.path.join(data_dir, data_filename)

    print("save to", data_file_path)

    dataset = get_dataset(is_test=args.is_test, is_denoise=args.is_denoise)
    # dataset = "/media/guus/Secondary/Data_HeadNeck"
    print("reading folder:", dataset)

    if args.overwrite or not os.path.exists(data_file_path):
        training_files = fetch_training_data_files(dataset)
        write_data_to_file(training_files, data_file_path,
                           brats_dir=BRATS_DIR,
                           dataset=dataset,
                           config=config,
                           image_shape=get_shape_from_string(args.image_shape),
                           crop=str2bool(args.crop),
                           is_normalize=args.is_normalize,
                           is_hist_match=args.is_hist_match,
                           is_denoise=args.is_denoise)


def main():
    args = get_args.prepare_data_headneck()

    args.is_test = "0"
    for is_denoise in ["0"]:
        args.is_denoise = is_denoise
        for is_normalize in ["z"]:
            args.is_normalize = is_normalize
            for is_hist_match in ["0"]:
                args.is_hist_match = is_hist_match

                print(">> prepare data {} {} {}".format(
                    is_denoise, is_normalize, is_hist_match))
                prepare_data(args)


if __name__ == "__main__":
    main()
    print_separator()
