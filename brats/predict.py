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


core_name = "brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-1"
core_name = "brats_2018_is-160-192-128_crop-1_bias-1_denoise-bm4d_norm-01_hist-1"
core_name = "brats_2018_is-160-192-128_crop-1_bias-1_denoise-bm4d_norm-01_hist-1"

model_name = "ps-128-128-128_unet_crf-0_d-4_nb-16_loss-weighted_model"
# model_name = "ps-128-128-128_isensee_crf-0_loss-weighted_model"
# model_name = "ps-64-64-64_densefcn_crf-0_loss-weighted_model"
# model_name = "ps-64-64-64_densenfcn_crf-0_loss-tv_minh_model"

config["data_file"] = os.path.abspath(
    "brats/database/data/{}_data.h5".format(core_name))

config["model_file"] = os.path.abspath(
    "/home/minhvu/github/3DUnetCNN_BRATS/brats/database/model/base/{}_{}.h5".format(core_name, model_name))
config["model_file"] = os.path.abspath(
    "/home/minhvu/github/3DUnetCNN_BRATS/brats/database/model/{}_{}.h5".format(core_name, model_name))
config["model_file"] = os.path.abspath(
    "/home/minhvu/github/3DUnetCNN_BRATS/brats/database/model/finetune/{}_{}.h5".format(core_name, model_name))

config["training_file"] = os.path.abspath(
    "brats/database/train_ids/{}_train_ids.h5".format(core_name))
config["validation_file"] = os.path.abspath(
    "brats/database/valid_ids/{}_valid_ids.h5".format(core_name))
config["prediction_folder"] = os.path.abspath(
    "brats/database/prediction/{}_{}".format(core_name, model_name))


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
