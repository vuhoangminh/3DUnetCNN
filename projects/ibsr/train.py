from comet_ml import Experiment

import os

from unet3d.data import open_data_file
from unet3d.generator import get_training_and_validation_and_testing_generators
from unet3d.model import unet_model_3d, simple_model_3d, eye_model_3d, mnet_model_3d, multiscale_unet_model_3d
from unet3d.model import isensee2017_model
from unet3d.model import densefcn_model_3d
from unet3d.model import dense_unet_3d, res_unet_3d, se_unet_3d
from unet3d.training import train_model
from unet3d.utils.path_utils import get_project_dir
from unet3d.utils.path_utils import get_shape_from_string
from unet3d.utils.path_utils import get_training_h5_paths
from unet3d.utils.path_utils import make_dir
import unet3d.utils.args_utils as get_args

from brats.prepare_data import prepare_data

from unet3d.utils.print_utils import print_section, print_separator

from brats.config import config, config_unet


config.update(config_unet)

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def train(overwrite=True, crop=True, challenge="brats", year=2018,
          image_shape="160-160-128", is_bias_correction="1",
          is_normalize="z", is_denoise="0",
          is_hist_match="0", is_test="1",
          depth_unet=4, n_base_filters_unet=16, model_name="unet",
          patch_shape="128-128-128", is_crf="0",
          batch_size=1, loss="weighted",
          weight_tv_to_main_loss=0.1):

    data_path, trainids_path, validids_path, testids_path, model_path = get_training_h5_paths(
        brats_dir=BRATS_DIR, overwrite=overwrite, crop=crop, challenge=challenge, year=year,
        image_shape=image_shape, is_bias_correction=is_bias_correction,
        is_normalize=is_normalize, is_denoise=is_denoise,
        is_hist_match=is_hist_match, is_test=is_test,
        model_name=model_name, depth_unet=depth_unet, n_base_filters_unet=n_base_filters_unet,
        patch_shape=patch_shape, is_crf=is_crf, loss=loss, model_dim=3,
        weight_tv_to_main_loss=weight_tv_to_main_loss)

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
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_and_testing_generators(
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
    if not overwrite and os.path.exists(config["model_file"]):
        print("load old model")
        from unet3d.utils.model_utils import generate_model
        model = generate_model(config["model_file"], loss_function=loss)
        # model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        if model_name == "unet":
            print("init unet model")
            model = unet_model_3d(input_shape=config["input_shape"],
                                  pool_size=config["pool_size"],
                                  n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  deconvolution=config["deconvolution"],
                                  depth=depth_unet,
                                  n_base_filters=n_base_filters_unet,
                                  loss_function=loss)

        # elif model_name == "densefcn":
        #     print("init densenet model")
        #     # config["initial_learning_rate"] = 1e-5
        #     model = densefcn_model_3d(input_shape=config["input_shape"],
        #                               classes=config["n_labels"],
        #                               initial_learning_rate=config["initial_learning_rate"],
        #                               nb_dense_block=5,
        #                               nb_layers_per_block=4,
        #                               early_transition=True,
        #                               dropout_rate=0.2,
        #                               loss_function=loss)

        # elif model_name == "denseunet":
        #     print("init denseunet model")
        #     model = dense_unet_3d(input_shape=config["input_shape"],
        #                           pool_size=config["pool_size"],
        #                           n_labels=config["n_labels"],
        #                           initial_learning_rate=config["initial_learning_rate"],
        #                           deconvolution=config["deconvolution"],
        #                           depth=depth_unet,
        #                           n_base_filters=n_base_filters_unet,
        #                           loss_function=loss)

        # elif model_name == "resunet":
        #     print("init resunet model")
        #     model = res_unet_3d(input_shape=config["input_shape"],
        #                         pool_size=config["pool_size"],
        #                         n_labels=config["n_labels"],
        #                         initial_learning_rate=config["initial_learning_rate"],
        #                         deconvolution=config["deconvolution"],
        #                         depth=depth_unet,
        #                         n_base_filters=n_base_filters_unet,
        #                         loss_function=loss)

        # elif model_name == "seunet":
        #     print("init seunet model")
        #     model = se_unet_3d(input_shape=config["input_shape"],
        #                        pool_size=config["pool_size"],
        #                        n_labels=config["n_labels"],
        #                        initial_learning_rate=config["initial_learning_rate"],
        #                        deconvolution=config["deconvolution"],
        #                        depth=depth_unet,
        #                        n_base_filters=n_base_filters_unet,
        #                        loss_function=loss)

        # elif model_name == "simple":
        #     print("init simple model")
        #     model = simple_model_3d(input_shape=config["input_shape"],
        #                             pool_size=config["pool_size"],
        #                             n_labels=config["n_labels"],
        #                             initial_learning_rate=config["initial_learning_rate"],
        #                             depth=depth_unet,
        #                             n_base_filters=n_base_filters_unet,
        #                             loss_function=loss)

        # elif model_name == "eye":
        #     print("init eye model")
        #     model = eye_model_3d(input_shape=config["input_shape"],
        #                          pool_size=config["pool_size"],
        #                          n_labels=config["n_labels"],
        #                          initial_learning_rate=config["initial_learning_rate"],
        #                          depth=depth_unet,
        #                          n_base_filters=n_base_filters_unet,
        #                          growth_rate=4,
        #                          loss_function=loss)

        # elif model_name == "m":
        #     print("init mnet model")
        #     model = mnet_model_3d(input_shape=config["input_shape"],
        #                           pool_size=config["pool_size"],
        #                           n_labels=config["n_labels"],
        #                           initial_learning_rate=config["initial_learning_rate"],
        #                           n_base_filters=32,
        #                           loss_function=loss)

        # elif model_name == "m2":
        #     print("init mnet model")
        #     from unet3d.model.mnet import mnet_model2_3d
        #     model = mnet_model2_3d(input_shape=config["input_shape"],
        #                            pool_size=config["pool_size"],
        #                            n_labels=config["n_labels"],
        #                            initial_learning_rate=config["initial_learning_rate"],
        #                            n_base_filters=16,
        #                            loss_function=loss)

        elif model_name == "multi":
            print("init multiscale unet model")
            model = multiscale_unet_model_3d(input_shape=config["input_shape"],
                                             pool_size=config["pool_size"],
                                             n_labels=config["n_labels"],
                                             initial_learning_rate=config["initial_learning_rate"],
                                             deconvolution=config["deconvolution"],
                                             depth=depth_unet,
                                             n_base_filters=n_base_filters_unet,
                                             loss_function=loss)

        else:
            print("init isensee model")
            model = isensee2017_model(input_shape=config["input_shape"],
                                      n_labels=config["n_labels"],
                                      initial_learning_rate=config["initial_learning_rate"],
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
    print("data file:", config["data_file"])
    print("model file:", config["model_file"])
    print("training file:", config["training_file"])
    print("validation file:", config["validation_file"])
    print("testing file:", config["testing_file"])

    if is_test == "1":
        config["n_epochs"] = 5

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
    from keras import backend as K
    K.clear_session()


def main():
    args = get_args.train()
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
    weight_tv_to_main_loss = args.weight_tv_to_main_loss

    train(overwrite=overwrite, crop=crop, challenge=challenge, year=year,
          image_shape=image_shape, is_bias_correction=is_bias_correction,
          is_normalize=is_normalize, is_denoise=is_denoise,
          is_hist_match=is_hist_match, is_test=is_test,
          model_name=model_name, depth_unet=depth_unet, n_base_filters_unet=n_base_filters_unet,
          patch_shape=patch_shape, is_crf=is_crf, batch_size=batch_size,
          loss=loss, weight_tv_to_main_loss=weight_tv_to_main_loss)


if __name__ == "__main__":
    main()
