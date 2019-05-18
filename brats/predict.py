import os
import glob
import gc
from unet3d.utils import pickle_load

import unet3d.utils.args_utils as get_args
from unet3d.utils.path_utils import get_project_dir
from unet3d.utils.path_utils import get_training_h5_paths
from unet3d.utils.path_utils import get_shape_from_string
from unet3d.utils.path_utils import get_filename_without_extension
from unet3d.utils.path_utils import make_dir

from brats.config import config, config_unet, config_dict
config.update(config_unet)


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def is_all_cases_predicted(prediction_folder, testing_file):
    data_file = pickle_load(config["testing_file"])
    num_cases = len(data_file)
    if not os.path.exists(prediction_folder):
        return False
    else:
        num_predicted = len(
            glob.glob(os.path.join(config["prediction_folder"], "*")))
        return num_cases == num_predicted


list_already_predicted = list()


def predict(args, prediction_dir="desktop"):

    data_path, trainids_path, validids_path, testids_path, model_path = get_training_h5_paths(
        brats_dir=BRATS_DIR, args=args)

    if not os.path.exists(model_path):
        print("model not exists. Please check")
    else:
        config["data_file"] = data_path
        config["model_file"] = model_path
        config["training_file"] = trainids_path
        config["validation_file"] = validids_path
        config["testing_file"] = testids_path
        config["patch_shape"] = get_shape_from_string(args.patch_shape)
        config["input_shape"] = tuple(
            [config["nb_channels"]] + list(config["patch_shape"]))

        if prediction_dir == "SERVER":
            prediction_dir = "brats"
        else:
            prediction_dir = "/mnt/sda/3DUnetCNN_BRATS/brats"

        config["prediction_folder"] = os.path.join(
            prediction_dir, "database/prediction", get_filename_without_extension(config["model_file"]))

        if is_all_cases_predicted(config["prediction_folder"], config["testing_file"]):
            print("Already predicted. Skip...")
            list_already_predicted.append(config["prediction_folder"])
        else:
            make_dir(config["prediction_folder"])

            print("-"*60)
            print("SUMMARY")
            print("-"*60)
            print("data file:", config["data_file"])
            print("model file:", config["model_file"])
            print("training file:", config["training_file"])
            print("validation file:", config["validation_file"])
            print("testing file:", config["testing_file"])
            print("prediction folder:", config["prediction_folder"])
            print("-"*60)

            if not os.path.exists(config["model_file"]):
                raise ValueError(
                    "can not find model {}. Please check".format(config["model_file"]))

            if args.model_dim == 3:
                from unet3d.prediction import run_validation_cases
            elif args.model_dim == 25:
                from unet25d.prediction import run_validation_cases
            elif args.model_dim == 2:
                from unet2d.prediction import run_validation_cases
            else:
                raise ValueError(
                    "dim {} NotImplemented error. Please check".format(args.model_dim))

            run_validation_cases(validation_keys_file=config["testing_file"],
                                 model_file=config["model_file"],
                                 training_modalities=config["training_modalities"],
                                 labels=config["labels"],
                                 hdf5_file=config["data_file"],
                                 output_label_map=True,
                                 output_dir=config["prediction_folder"])


def main():
    args = get_args.train25d()

    depth_unet = args.depth_unet
    n_base_filters_unet = args.n_base_filters_unet
    patch_shape = args.patch_shape
    is_crf = args.is_crf
    batch_size = args.batch_size
    is_hist_match = args.is_hist_match
    loss = args.loss

    header = ("dice_WholeTumor", "dice_TumorCore", "dice_EnhancingTumor")
    model_scores = list()
    model_ids = list()

    for is_augment in ["1"]:
        args.is_augment = is_augment
        for model_name in ["casnet_v2"]:
            args.model = model_name
            for is_denoise in ["0"]:
                args.is_denoise = is_denoise
                for is_normalize in ["z"]:
                    args.is_normalize = is_normalize
                    for is_hist_match in ["0"]:
                        args.is_hist_match = is_hist_match
                        for loss in ["weighted"]:
                            args.loss = loss
                            for patch_shape in ["160-192-1"]:
                                args.patch_shape = patch_shape
                                args.model_dim = 2

                                print("="*120)
                                print(
                                    ">> processing model-{}{}, depth-{}, filters-{}, patch_shape-{}, is_denoise-{}, is_normalize-{}, is_hist_match-{}, loss-{}".format(
                                        model_name,
                                        args.model_dim,
                                        depth_unet,
                                        n_base_filters_unet,
                                        patch_shape,
                                        is_denoise,
                                        is_normalize,
                                        is_hist_match,
                                        loss))
                                is_test = "0"
                                predict(args)
                                # print("="*60)
                                print(">> finished")
                                print("="*120)
                                gc.collect()
                                from keras import backend as K
                                K.clear_session()

    print(list_already_predicted)
    print(len(list_already_predicted))


if __name__ == "__main__":
    main()
