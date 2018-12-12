from unet3d.utils.path_utils import get_project_dir
import os

from unet3d.prediction import run_validation_cases

from brats.config import config, config_unet
config.update(config_unet)


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])

config["data_file"] = os.path.abspath(
    "brats/database/brats2018_test_normalize_minh/brats_data.h5")
config["model_file"] = os.path.abspath(
    "brats/database/brats2018_test_normalize_minh/tumor_segmentation_model.h5")
config["training_file"] = os.path.abspath(
    "brats/database/brats2018_test_normalize_minh/training_ids.pkl")
config["validation_file"] = os.path.abspath(
    "brats/database/brats2018_test_normalize_minh/validation_ids.pkl")
config["n_steps_file"] = os.path.abspath(
    "brats/database/brats2018_test_normalize_minh/n_step.pkl")
config["prediction_folder"] = os.path.abspath(
    "brats/database/brats2018_test_normalize_minh/prediction")

def main():
    prediction_dir = os.path.abspath(config["prediction_folder"])
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    main()
