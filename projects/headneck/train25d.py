from comet_ml import Experiment

# to compute memory consumption ----------------------------------
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config_tf = tf.ConfigProto()
# config_tf.gpu_options.per_process_gpu_memory_fraction = 0.015
# config_tf.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config_tf))
# to compute memory consumption ----------------------------------

import os

from unet3d.data import open_data_file
from unet25d.generator import get_training_and_validation_and_testing_generators25d
from unet25d.model import *
from unet3d.training import train_model
from unet3d.utils.path_utils import get_project_dir
from unet3d.utils.path_utils import get_shape_from_string
from unet3d.utils.path_utils import get_training_h5_paths
import unet3d.utils.args_utils as get_args
import unet3d.utils.path_utils as path_utils

from brats.prepare_data import prepare_data

from unet3d.utils.print_utils import print_section

from brats.config import config, config_unet

# pp = pprint.PrettyPrinter(indent=4)
# # pp.pprint(config)
config.update(config_unet)
# pp.pprint(config)


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def fix_learning_rate():
    if config["patch_shape"] in ["160-192-13", "160-192-15", "160-192-17"]:
        config["initial_learning_rate"] = 1e-4
    if config["patch_shape"] in ["160-192-3"]:
        config["initial_learning_rate"] = 1e-2
    return config


def train(args):

    data_path, trainids_path, validids_path, testids_path, model_path = get_training_h5_paths(
        brats_dir=BRATS_DIR, args=args)

    config["data_file"] = data_path
    config["model_file"] = model_path
    config["training_file"] = trainids_path
    config["validation_file"] = validids_path
    config["testing_file"] = testids_path
    config["patch_shape"] = get_shape_from_string(args.patch_shape)
    config["input_shape"] = tuple(
        [config["nb_channels"]] + list(config["patch_shape"]))

    if args.patch_shape in ["160-192-13", "160-192-15", "160-192-17"]:
        config["initial_learning_rate"] = 1e-4
        print("lr updated...")
    if args.patch_shape in ["160-192-3"]:
        config["initial_learning_rate"] = 1e-2
        print("lr updated...")

    if args.overwrite or not os.path.exists(data_path):
        prepare_data(args)

    print_section("Open file")
    data_file_opened = open_data_file(config["data_file"])

    print_section("get training and testing generators")
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_and_testing_generators25d(
        data_file_opened,
        batch_size=args.batch_size,
        data_split=config["validation_split"],
        overwrite=args.overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        testing_keys_file=config["testing_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=args.batch_size,
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        augment_flipud=config["augment_flipud"],
        augment_fliplr=config["augment_fliplr"],
        augment_elastic=config["augment_elastic"],
        augment_rotation=config["augment_rotation"],
        augment_shift=config["augment_shift"],
        augment_shear=config["augment_shear"],
        augment_zoom=config["augment_zoom"],
        n_augment=config["n_augment"],
        skip_blank=config["skip_blank"],
        is_test=args.is_test)

    print("-"*60)
    print("# Load or init model")
    print("-"*60)

    if not args.overwrite and os.path.exists(config["model_file"]):
        print("load old model")
        from unet3d.utils.model_utils import generate_model
        model = generate_model(config["model_file"], loss_function=args.loss)
        # model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        if args.model == "seunet":
            print("init seunet model")
            model = unet_model_25d(input_shape=config["input_shape"],
                                   n_labels=config["n_labels"],
                                   initial_learning_rate=config["initial_learning_rate"],
                                   deconvolution=config["deconvolution"],
                                   #   batch_normalization=True,
                                   depth=args.depth_unet,
                                   n_base_filters=args.n_base_filters_unet,
                                   loss_function=args.loss,
                                   is_unet_original=False)
        elif args.model == "unet":
            print("init unet model")
            model = unet_model_25d(input_shape=config["input_shape"],
                                   n_labels=config["n_labels"],
                                   initial_learning_rate=config["initial_learning_rate"],
                                   deconvolution=config["deconvolution"],
                                   #   batch_normalization=True,
                                   depth=args.depth_unet,
                                   n_base_filters=args.n_base_filters_unet,
                                   loss_function=args.loss)
        elif args.model == "segnet":
            print("init segnet model")
            model = segnet25d(input_shape=config["input_shape"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet,
                              loss_function=args.loss)
        elif args.model == "isensee":
            print("init isensee model")
            model = isensee25d_model(input_shape=config["input_shape"],
                                     n_labels=config["n_labels"],
                                     initial_learning_rate=config["initial_learning_rate"],
                                     loss_function=args.loss)

    model.summary()

    print("-"*60)
    print("# start training")
    print("-"*60)
    # run training

    if args.is_test == "0":
        experiment = Experiment(api_key="34T3kJ5CkXUtKAbhI6foGNFBL",
                                project_name="train",
                                workspace="guusgrimbergen")
    else:
        experiment = None

    print(config["initial_learning_rate"], config["learning_rate_drop"])
    train_model(experiment=experiment,
                model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"]
                )

    if args.is_test == "0":
        experiment.log_parameters(config)

    data_file_opened.close()
    from keras import backend as K
    K.clear_session()


def main():
    global config
    args = get_args.train25d()

    config = path_utils.update_is_augment(args, config)

    data_path, _, _, _, _ = path_utils.get_training_h5_paths(BRATS_DIR, args)
    if args.overwrite or not os.path.exists(data_path):
        prepare_data(args)

    train(args)


if __name__ == "__main__":
    main()
