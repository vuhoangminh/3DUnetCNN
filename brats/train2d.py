from comet_ml import Experiment

import os
import glob
import pprint
import numpy as np

from unet3d.data import write_data_to_file, open_data_file
from unet2d.generator import get_training_and_validation_and_testing_generators2d
from unet2d.model import unet_model_2d
from unet3d.training import load_old_model, train_model
from unet3d.utils.path_utils import get_project_dir, get_h5_training_dir, get_model_h5_filename
from unet3d.utils.path_utils import get_training_h5_filename, get_shape_string, get_shape_from_string
from unet3d.utils.path_utils import get_training_h5_paths
import unet3d.utils.args_utils as get_args

from brats.prepare_data import prepare_data

from unet3d.utils.print_utils import print_processing, print_section, print_separator

from brats.config import config, config_unet

# pp = pprint.PrettyPrinter(indent=4)
# # pp.pprint(config)
config.update(config_unet)
# pp.pprint(config)


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def make_dir(data_dir):
    # make dir
    if not os.path.exists(data_dir):
        print_separator()
        print("making dir", data_dir)
        os.makedirs(data_dir)


def train(overwrite=True, crop=True, challenge="brats", year=2018,
          image_shape="160-160-128", is_bias_correction="1",
          is_normalize="z", is_denoise="0",
          is_hist_match="0", is_test="1",
          depth_unet=4, n_base_filters_unet=16, model_name="unet",
          patch_shape="128-128-128", is_crf="0",
          batch_size=1, loss="weighted"):

    data_path, trainids_path, validids_path, testids_path, model_path = get_training_h5_paths(
        brats_dir=BRATS_DIR, overwrite=overwrite, crop=crop, challenge=challenge, year=year,
        image_shape=image_shape, is_bias_correction=is_bias_correction,
        is_normalize=is_normalize, is_denoise=is_denoise,
        is_hist_match=is_hist_match, is_test=is_test,
        model_name=model_name, depth_unet=depth_unet, n_base_filters_unet=n_base_filters_unet,
        patch_shape=patch_shape, is_crf=is_crf, loss=loss, model_dim=2)

    config["data_file"] = data_path
    config["model_file"] = model_path
    config["training_file"] = trainids_path
    config["validation_file"] = validids_path
    config["testing_file"] = testids_path
    config["patch_shape"] = get_shape_from_string(patch_shape)
    config["input_shape"] = tuple(
        [config["nb_channels"]] + list(config["patch_shape"]))

    if overwrite or not os.path.exists(data_path):
        prepare_data(overwrite=overwrite, crop=crop, challenge=challenge, year=year,
                     image_shape=image_shape, is_bias_correction=is_bias_correction,
                     is_normalize=is_normalize, is_denoise=is_denoise,
                     is_hist_match=is_hist_match, is_test=is_test)

    print_section("Open file")
    data_file_opened = open_data_file(config["data_file"])

    print_section("get training and testing generators")
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_and_testing_generators2d(
        data_file_opened,
        batch_size=batch_size,
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        testing_keys_file=config["testing_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=batch_size,
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
        skip_blank=config["skip_blank"])

    print("-"*60)
    print("# Load or init model")
    print("-"*60)
    config["input_shape"]=config["input_shape"][0:len(config["input_shape"])-1]
    if not overwrite and os.path.exists(config["model_file"]):
        print("load old model")
        from unet3d.utils.model_utils import generate_model
        model = generate_model(config["model_file"])
        # model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        # if model_name == "unet":
        print("init unet model")
        model = unet_model_2d(input_shape=config["input_shape"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=depth_unet,
                              n_base_filters=n_base_filters_unet,
                              loss_function=loss)

    model.summary()

    print("-"*60)
    print("# start training")
    print("-"*60)
    # run training

    if is_test == "0":
        experiment = Experiment(api_key="AgTGwIoRULRgnfVR5M8mZ5AfS",
                                project_name="train",
                                workspace="vuhoangminh")
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

    if is_test == "0":
        experiment.log_parameters(config)

    data_file_opened.close()


def main():
    args = get_args.train2d()
    overwrite = args.overwrite
    crop = args.crop
    challenge = args.challenge
    year = args.year
    image_shape = args.image_shape
    is_bias_correction = args.is_bias_correction
    is_normalize = args.is_normalize
    is_denoise = args.is_denoise
    is_test = args.is_test
    model_name = args.model
    depth_unet = args.depth_unet
    n_base_filters_unet = args.n_base_filters_unet
    patch_shape = args.patch_shape
    is_crf = args.is_crf
    batch_size = args.batch_size
    is_hist_match = args.is_hist_match
    loss = args.loss

    train(overwrite=overwrite, crop=crop, challenge=challenge, year=year,
          image_shape=image_shape, is_bias_correction=is_bias_correction,
          is_normalize=is_normalize, is_denoise=is_denoise,
          is_hist_match=is_hist_match, is_test=is_test,
          model_name=model_name, depth_unet=depth_unet, n_base_filters_unet=n_base_filters_unet,
          patch_shape=patch_shape, is_crf=is_crf, batch_size=batch_size,
          loss=loss)


if __name__ == "__main__":
    main()
