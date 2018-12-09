import os
import glob
import pprint

from unet3d.data import open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.utils.print_utils import print_processing, print_section, print_separator

from brats.config import config, config_unet


config["data_file"] = os.path.abspath("brats/database/brats2018_test_normalize_minh/brats_data.h5")
config["model_file"] = os.path.abspath("brats/database/brats2018_test_normalize_minh/tumor_segmentation_model.h5")
config["training_file"] = os.path.abspath("brats/database/brats2018_test_normalize_minh/training_ids.pkl")
config["validation_file"] = os.path.abspath("brats/database/brats2018_test_normalize_minh/validation_ids.pkl")
config["n_steps_file"] = os.path.abspath("brats/database/brats2018_test_normalize_minh/n_step.pkl")


def main(overwrite=False):
    print_section("Open file")
    data_file_opened = open_data_file(config["data_file"])


    print_section("get training and testing generators")    
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_steps_file=config["n_steps_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])



if __name__ == "__main__":
    main(False)
    # print_separator()

