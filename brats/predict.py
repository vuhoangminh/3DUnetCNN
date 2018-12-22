from unet3d.utils.path_utils import get_project_dir
import os

from unet3d.prediction import run_validation_cases

from brats.config import config, config_unet
config.update(config_unet)


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])

# config["data_file"] = os.path.abspath(
#     "brats/database/data/brats_2018_is-160-192-128_crop-1_bias-1_denoise-bm4d_norm-01_hist-1_data.h5")
# config["model_file"] = os.path.abspath(
#     "brats/database/model/done/minh_loss/brats_2018_is-160-192-128_crop-1_bias-1_denoise-bm4d_norm-01_hist-1_ps-128-128-128_unet_crf-0_d-4_nb-16_model.h5")
# config["training_file"] = os.path.abspath(
#     "brats/database/train_ids/brats_2018_is-160-192-128_crop-1_bias-1_denoise-bm4d_norm-01_hist-1_train_ids.h5")
# config["validation_file"] = os.path.abspath(
#     "brats/database/valid_ids/brats_2018_is-160-192-128_crop-1_bias-1_denoise-bm4d_norm-01_hist-1_valid_ids.h5")
# config["prediction_folder"] = os.path.abspath(
#     "brats/database/prediction/brats_2018_is-160-192-128_crop-1_bias-1_denoise-bm4d_norm-01_hist-1_ps-128-128-128_unet_crf-0_d-4_nb-16_model")


config["data_file"] = os.path.abspath(
    "brats/database/data/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_data.h5")
config["model_file"] = os.path.abspath(
    "brats/database/model/done/weighted_loss/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_ps-128-128-128_isensee_crf-0_model.h5")
config["training_file"] = os.path.abspath(
    "brats/database/train_ids/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_train_ids.h5")
config["validation_file"] = os.path.abspath(
    "brats/database/valid_ids/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_valid_ids.h5")
config["prediction_folder"] = os.path.abspath(
    "brats/database/prediction/brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_ps-128-128-128_isensee_crf-0_model")


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
