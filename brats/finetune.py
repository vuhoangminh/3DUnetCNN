from comet_ml import Experiment

import os

from unet3d.data import open_data_file
from unet3d.training import train_model
from unet3d.utils.path_utils import get_project_dir
from unet3d.utils.path_utils import get_shape_from_string
from unet3d.utils.path_utils import make_dir
from unet3d.utils.path_utils import get_model_baseline_path
from unet3d.utils.path_utils import get_training_h5_paths
import unet3d.utils.args_utils as get_args
import unet3d.utils.path_utils as path_utils

from brats.prepare_data import prepare_data
from unet3d.utils.print_utils import print_section
from brats.config import config, config_unet, config_finetune

# pp = pprint.PrettyPrinter(indent=4)
# # pp.pprint(config)
config.update(config_unet)
# pp.pprint(config)

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def finetune(args):

    data_path, trainids_path, validids_path, testids_path, model_path = get_training_h5_paths(
        brats_dir=BRATS_DIR, args=args)

    if args.name != "0":
        model_path = args.name

    config["data_file"] = data_path
    config["model_file"] = model_path
    config["training_file"] = trainids_path
    config["validation_file"] = validids_path
    config["testing_file"] = testids_path
    config["patch_shape"] = get_shape_from_string(args.patch_shape)
    config["input_shape"] = tuple(
        [config["nb_channels"]] + list(config["patch_shape"]))

    if args.overwrite or not os.path.exists(data_path):
        prepare_data(args)

    folder = os.path.join(BRATS_DIR, "database", "model", "base")

    if not os.path.exists(config["model_file"]):
        model_baseline_path = get_model_baseline_path(folder=folder, args=args)
        if model_baseline_path is None:
            raise ValueError("can not fine baseline model. Please check")
        else:
            config["model_file"] = model_baseline_path

    print_section("Open file")
    data_file_opened = open_data_file(config["data_file"])

    make_dir(config["training_file"])

    print_section("get training and testing generators")
    if args.model_dim == 3:
        from unet3d.generator import get_training_and_validation_and_testing_generators
        train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_and_testing_generators(
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
            is_create_patch_index_list_original=config["is_create_patch_index_list_original"],
            augment_flipud=config["augment_flipud"],
            augment_fliplr=config["augment_fliplr"],
            augment_elastic=config["augment_elastic"],
            augment_rotation=config["augment_rotation"],
            augment_shift=config["augment_shift"],
            augment_shear=config["augment_shear"],
            augment_zoom=config["augment_zoom"],
            n_augment=config["n_augment"],
            skip_blank=config["skip_blank"])
    elif args.model_dim == 25:
        from unet25d.generator import get_training_and_validation_and_testing_generators25d
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
    else:
        from unet2d.generator import get_training_and_validation_and_testing_generators2d
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
            is_test=args.is_test)

    print("-"*60)
    print("# Load or init model")
    print("-"*60)
    print(">> update config file")
    config.update(config_finetune)
    if not os.path.exists(config["model_file"]):
        raise Exception(
            "{} model file not found. Please try again".format(config["model_file"]))
    else:
        from unet3d.utils.model_utils import generate_model
        print(">> load old and generate model")
        model = generate_model(config["model_file"],
                               initial_learning_rate=config["initial_learning_rate"],
                               loss_function=args.loss,
                               weight_tv_to_main_loss=args.weight_tv_to_main_loss)
        model.summary()

    # run training
    print("-"*60)
    print("# start finetuning")
    print("-"*60)

    print("Number of training steps: ", n_train_steps)
    print("Number of validation steps: ", n_validation_steps)

    data_path, trainids_path, validids_path, testids_path, model_path = get_training_h5_paths(
        brats_dir=BRATS_DIR,
        args=args,
        dir_read_write="finetune",
        is_finetune=True)

    config["model_file"] = model_path

    if os.path.exists(config["model_file"]):
        print("{} existed. Will skip!!!".format(config["model_file"]))
    else:

        if args.is_test == "1":
            config["n_epochs"] = 5

        if args.is_test == "0":
            experiment = Experiment(api_key="AgTGwIoRULRgnfVR5M8mZ5AfS",
                                    project_name="finetune",
                                    workspace="vuhoangminh")
        else:
            experiment = None

        if args.model_dim == 2 and args.model == "isensee":
            config["initial_learning_rate"] = 1e-7

        print(config["initial_learning_rate"], config["learning_rate_drop"])
        print("data file:", config["data_file"])
        print("model file:", config["model_file"])
        print("training file:", config["training_file"])
        print("validation file:", config["validation_file"])
        print("testing file:", config["testing_file"])

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
    args = get_args.finetune()

    config = path_utils.update_is_augment(args, config)

    data_path, _, _, _, _ = path_utils.get_training_h5_paths(BRATS_DIR, args)
    if args.overwrite or not os.path.exists(data_path):
        prepare_data(args)

    finetune(args)


if __name__ == "__main__":
    main()
