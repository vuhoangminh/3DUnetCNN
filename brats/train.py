import os
import glob
import pprint
import numpy as np

from comet_ml import Experiment

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import unet_model_3d
from unet3d.training import load_old_model, train_model
from unet3d.utils.path_utils import get_brats_data_h5_path
import unet3d.utils.args_utils as get_args

from unet3d.utils.print_utils import print_processing, print_section, print_separator

from brats.config import config, config_unet

# pp = pprint.PrettyPrinter(indent=4)
# # pp.pprint(config)
config.update(config_unet)
# pp.pprint(config)


# experiment = Experiment(api_key="Nh9odbzbndSjh2N15O2S3d3fP",
#                         project_name="general", workspace="vuhoangminh")


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


def main(overwrite=False):
    args = get_args.train()
    overwrite = args.overwrite

    # config["data_file"] = get_brats_data_h5_path(args.competition, args.year,
    #                                              args.inputshape, args.isbiascorrection,
    #                                              args.normalization, args.clahe,
    #                                              args.histmatch)

    # print(config["data_file"])

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
        is_create_patch_index_list_original=config["is_create_patch_index_list_original"],
        augment_flipud=config["augment_flipud"], 
        augment_fliplr=config["augment_fliplr"], 
        augment_elastic=config["augment_elastic"],
        augment_rotation=config["augment_rotation"], 
        augment_shift=config["augment_shift"], 
        augment_shear=config["augment_shear"],
        augment_zoom=config["augment_zoom"], 
        n_augment=config["n_augment"])

    print("-"*60)
    print("# Load or init model")
    print("-"*60)
    if not overwrite and os.path.exists(config["model_file"]):
        print("load old model")
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        print("init model model")
        model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              depth=config["depth"],
                              n_base_filters=config["n_base_filters"])

    model.summary()

    # import nibabel as nib
    # temp_in_path = "C:/Users/minhm/Desktop/temp/template.nii.gz"
    # temp_out_path = "C:/Users/minhm/Desktop/temp/out.nii.gz"
    # temp_out_truth_path = "C:/Users/minhm/Desktop/temp/truth.nii.gz"

    # n_validation_samples = 0
    # validation_samples = list()
    # for i in range(10):
    #     x, y = next(validation_generator)
    #     hash_x = hash(str(x))
    #     validation_samples.append(hash_x)
    #     n_validation_samples += x.shape[0]

    #     temp_in = nib.load(temp_in_path)
    #     temp_out = nib.Nifti1Image(x[0][0], affine=temp_in.affine)
    #     nib.save(temp_out, temp_out_path)

    #     temp_out = nib.Nifti1Image(y[0][0], affine=temp_in.affine)
    #     nib.save(temp_out, temp_out_truth_path)

    # print(n_validation_samples)


    print("-"*60)
    print("# start training")
    print("-"*60)
    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    data_file_opened.close()

if __name__ == "__main__":
    main(False)
    # print_separator()
