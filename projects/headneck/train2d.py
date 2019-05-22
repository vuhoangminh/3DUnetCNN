from comet_ml import Experiment

# to compute memory consumption ----------------------------------
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config_tf = tf.ConfigProto()
# config_tf.gpu_options.per_process_gpu_memory_fraction = 0.015
# config_tf.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config_tf))
# to compute memory consumption ----------------------------------


from brats.config import config, config_unet
from unet3d.utils.print_utils import print_processing, print_section, print_separator
from brats.prepare_data import prepare_data
import unet3d.utils.args_utils as get_args
from unet3d.utils.path_utils import make_dir
from unet3d.utils.path_utils import get_training_h5_paths
from unet3d.utils.path_utils import get_training_h5_filename, get_shape_string, get_shape_from_string
from unet3d.utils.path_utils import get_project_dir, get_h5_training_dir, get_model_h5_filename
from unet3d.training import load_old_model, train_model
from unet2d.model import *
from unet2d.generator import get_training_and_validation_and_testing_generators2d
from unet3d.data import write_data_to_file, open_data_file
import numpy as np
import pprint
import glob
import os
import unet3d.utils.path_utils as path_utils
from unet3d.utils.utils import str2bool


# pp = pprint.PrettyPrinter(indent=4)
# # pp.pprint(config)
config.update(config_unet)
# pp.pprint(config)


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


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

    if "casnet" in args.model:
        config["data_type_generator"] = 'cascaded'
    elif "sepnet" in args.model:
        config["data_type_generator"] = 'separated'
    else:
        config["data_type_generator"] = 'combined'

    if args.overwrite or not os.path.exists(data_path):
        prepare_data(args)

    print_section("Open file")
    data_file_opened = open_data_file(config["data_file"])

    print_section("get training and testing generators")
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_and_testing_generators2d(
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
        is_test=args.is_test,
        data_type_generator=config["data_type_generator"])

    print("-"*60)
    print("# Load or init model")
    print("-"*60)
    config["input_shape"] = config["input_shape"][0:len(
        config["input_shape"])-1]
    if not args.overwrite and os.path.exists(config["model_file"]):
        print("load old model")
        from unet3d.utils.model_utils import generate_model
        if "casnet" in args.model:
            args.loss = "casweighted"
        model = generate_model(config["model_file"], loss_function=args.loss)
    else:
        # instantiate new model
        if args.model == "isensee":
            print("init isensee model")
            model = isensee2d_model(input_shape=config["input_shape"],
                                    n_labels=config["n_labels"],
                                    initial_learning_rate=config["initial_learning_rate"],
                                    loss_function=args.loss)
        elif args.model == "unet":
            print("init unet model")
            model = unet_model_2d(input_shape=config["input_shape"],
                                  n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  deconvolution=config["deconvolution"],
                                  depth=args.depth_unet,
                                  n_base_filters=args.n_base_filters_unet,
                                  loss_function=args.loss)
        elif args.model == "segnet":
            print("init segnet model")
            model = segnet2d(input_shape=config["input_shape"],
                             n_labels=config["n_labels"],
                             initial_learning_rate=config["initial_learning_rate"],
                             depth=args.depth_unet,
                             n_base_filters=args.n_base_filters_unet,
                             loss_function=args.loss)
        elif args.model == "casnet_v1":
            print("init casnet_v1 model")
            model = casnet_v1(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet,
                              loss_function="casweighted")
        elif args.model == "casnet_v2":
            print("init casnet_v2 model")
            model = casnet_v2(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet,
                              loss_function="casweighted")
        elif args.model == "casnet_v3":
            print("init casnet_v3 model")
            model = casnet_v3(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet,
                              loss_function="casweighted")
        elif args.model == "casnet_v4":
            print("init casnet_v4 model")
            model = casnet_v4(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet)
        elif args.model == "casnet_v5":
            print("init casnet_v5 model")
            model = casnet_v5(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet,
                              loss_function="casweighted")
        elif args.model == "casnet_v6":
            print("init casnet_v6 model")
            model = casnet_v6(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet)
        elif args.model == "casnet_v7":
            print("init casnet_v7 model")
            model = casnet_v7(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet)
        elif args.model == "casnet_v8":
            print("init casnet_v8 model")
            model = casnet_v8(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet)
        elif args.model == "sepnet_v1":
            print("init sepnet_v1 model")
            model = sepnet_v1(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet)
        elif args.model == "sepnet_v2":
            print("init sepnet_v2 model")
            model = sepnet_v2(input_shape=config["input_shape"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=args.depth_unet,
                              n_base_filters=args.n_base_filters_unet)

        else:
            raise ValueError("Model is NotImplemented. Please check")

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

    if args.model == "isensee":
        config["initial_learning_rate"] = 1e-6
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
    args = get_args.train2d()

    config = path_utils.update_is_augment(args, config)

    data_path, _, _, _, _ = path_utils.get_training_h5_paths(BRATS_DIR, args)
    if args.overwrite or not os.path.exists(data_path):
        prepare_data(args)

    train(args)


if __name__ == "__main__":
    main()
