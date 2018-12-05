import os
import glob
import argparse

from unet3d.generator import get_training_and_validation_generators
from unet3d.data import write_data_to_file, open_data_file
from unet3d.utils.print_utils import print_processing, print_section, print_separator

from brats.config import config
from unet3d.utils.path_utils import get_project_dir, get_h5_image_path

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def fetch_training_data_files(dataset):
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data_train", dataset, "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(
                subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files


def get_args():
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('-c', '--challenge', type=str,
                        choices=[2018], default=2018,
                        help="year of brats challenge")
    parser.add_argument('-d', '--dataset', type=str,
                        choices=config["dataset"],
                        default="test",
                        help="dataset type")
    parser.add_argument('-i', '--inms', type=bool,
                        default=False,
                        help="is normalize mean, standard deviation")
    parser.add_argument('-o', '--overwrite', type=bool,
                        default=True)                        
    args = parser.parse_args()
    return args


def main(overwrite=False):
    args = get_args()
    dataset = args.dataset
    is_normalize_mean_std = args.inms
    challenge = args.challenge
    overwrite = args.overwrite

    save_to_dir = get_h5_image_path(brats_dir=BRATS_DIR,
                                    is_normalize_mean_std=is_normalize_mean_std,
                                    challenge=challenge,
                                    dataset=dataset)

    # make dir
    if not os.path.exists(save_to_dir):
        print_separator()
        print("making dir", save_to_dir)
        os.makedirs(save_to_dir)

    print_section("convert input images into an hdf5 file")

    data_file_path = os.path.join(save_to_dir, "brats_data.h5")
    print("arguments", args)
    print("save to", data_file_path)
    if overwrite or not os.path.exists(data_file_path):
        training_files = fetch_training_data_files(dataset)
        write_data_to_file(training_files, data_file_path,
                           image_shape=(240, 240, 155),
                           brats_dir=BRATS_DIR,
                           crop=False,
                           is_normalize_mean_std=is_normalize_mean_std,
                           dataset=dataset
                           )


if __name__ == "__main__":
    main(False)
