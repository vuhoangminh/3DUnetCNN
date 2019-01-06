from comet_ml import Experiment

import os
import glob
import pprint
import numpy as np

from unet3d.data import write_data_to_file, open_data_file
from unet25d.generator import get_training_and_validation_and_testing_generators25d
from unet25d.model import unet_model_25d
from unet2d.generator import get_training_and_validation_and_testing_generators2d
from unet2d.model import unet_model_2d
from unet3d.training import load_old_model, train_model
from unet3d.utils.path_utils import get_project_dir, get_h5_training_dir, get_model_h5_filename
from unet3d.utils.path_utils import get_training_h5_filename, get_shape_string, get_shape_from_string
from unet3d.utils.path_utils import get_training_h5_paths
import unet3d.utils.args_utils as get_args
from unet25d.generator import get_number_of_patches, create_patch_index_list

from brats.prepare_data import prepare_data

from unet3d.utils.print_utils import print_processing, print_section, print_separator
from unet3d.utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data

from brats.config import config, config_unet

# pp = pprint.PrettyPrinter(indent=4)
# # pp.pprint(config)
config.update(config_unet)
# pp.pprint(config)


config["data_file"] = os.path.abspath(
    "brats/database/data/test_brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-1_data.h5")
config["training_file"] = os.path.abspath(
    "brats/database/train_val_test_ids/test_train_ids.h5")
config["validation_file"] = os.path.abspath(
    "brats/database/train_val_test_ids/test_valid_ids.h5")


def main(overwrite=False):
    args = get_args.train25d()
    overwrite = args.overwrite

    print_section("Open file")
    data_file_opened = open_data_file(config["data_file"])

    print_section("get training and testing generators")
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_and_testing_generators2d(
        data_file_opened,
        batch_size=1,
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        testing_keys_file=config["validation_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=(160, 192, 1),
        validation_batch_size=1,
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        augment_flipud=config["augment_flipud"],
        augment_fliplr=config["augment_fliplr"],
        augment_elastic=True,
        augment_rotation=config["augment_rotation"],
        augment_shift=config["augment_shift"],
        augment_shear=config["augment_shear"],
        augment_zoom=config["augment_zoom"],
        n_augment=config["n_augment"],
        skip_blank=config["skip_blank"])

    import nibabel as nib
    laptop_save_dir = "C:/Users/minhm/Desktop/temp/"
    desktop_save_dir = "/home/minhvu/Desktop/temp/"
    save_dir = laptop_save_dir
    temp_in_path = save_dir + "template.nii.gz"
    temp_out_path = save_dir + "out.nii.gz"
    temp_out_truth_path = save_dir + "truth.nii.gz"
    temp_out_truth_path2 = save_dir + "truth2.nii.gz"
    temp_out_truth_path3 = save_dir + "truth3.nii.gz"

    n_validation_samples = 0
    validation_samples = list()
    for i in range(200):
        print(i)
        x, y = next(train_generator)

        if i > 86 and i % 5 == 0:
            hash_x = hash(str(x))
            validation_samples.append(hash_x)
            n_validation_samples += x.shape[0]

            temp_in = nib.load(temp_in_path)
            temp_out = nib.Nifti1Image(x[0][0], affine=temp_in.affine)
            nib.save(temp_out, temp_out_path)

            temp_out = nib.Nifti1Image(y[0][0], affine=temp_in.affine)
            nib.save(temp_out, temp_out_truth_path)

            temp_out = nib.Nifti1Image(y[0][1], affine=temp_in.affine)
            nib.save(temp_out, temp_out_truth_path2)

            temp_out = nib.Nifti1Image(y[0][2], affine=temp_in.affine)
            nib.save(temp_out, temp_out_truth_path3)

    print(n_validation_samples)


def main25d(overwrite=False):
    args = get_args.train25d()
    overwrite = args.overwrite

    print_section("Open file")
    data_file_opened = open_data_file(config["data_file"])

    print_section("get training and testing generators")
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_and_testing_generators25d(
        data_file_opened,
        batch_size=1,
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        testing_keys_file=config["validation_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=(160, 192, 7),
        validation_batch_size=1,
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        augment_flipud=config["augment_flipud"],
        augment_fliplr=config["augment_fliplr"],
        augment_elastic=True,
        augment_rotation=config["augment_rotation"],
        augment_shift=config["augment_shift"],
        augment_shear=config["augment_shear"],
        augment_zoom=config["augment_zoom"],
        n_augment=config["n_augment"],
        skip_blank=config["skip_blank"])

    import nibabel as nib
    laptop_save_dir = "C:/Users/minhm/Desktop/temp/"
    desktop_save_dir = "/home/minhvu/Desktop/temp/"
    save_dir = laptop_save_dir
    temp_in_path = save_dir + "template.nii.gz"
    temp_out_path = save_dir + "out.nii.gz"
    temp_out_path2 = save_dir + "out2.nii.gz"
    temp_out_path3 = save_dir + "out3.nii.gz"
    temp_out_path4 = save_dir + "out4.nii.gz"
    temp_out_truth_path = save_dir + "truth.nii.gz"
    # temp_out_truth_path2 = save_dir + "truth2.nii.gz"
    # temp_out_truth_path3 = save_dir + "truth3.nii.gz"

    n_validation_samples = 0
    validation_samples = list()
    for i in range(200):
        print(i)
        x, y = next(train_generator)

        if i > 86 and i % 5 == 0:
            hash_x = hash(str(x))
            validation_samples.append(hash_x)
            n_validation_samples += x.shape[0]

            temp_in = nib.load(temp_in_path)
            temp_out = nib.Nifti1Image(x[0][0], affine=temp_in.affine)
            nib.save(temp_out, temp_out_path)

            temp_in = nib.load(temp_in_path)
            temp_out = nib.Nifti1Image(x[0][1], affine=temp_in.affine)
            nib.save(temp_out, temp_out_path2)

            temp_in = nib.load(temp_in_path)
            temp_out = nib.Nifti1Image(x[0][2], affine=temp_in.affine)
            nib.save(temp_out, temp_out_path3)

            temp_in = nib.load(temp_in_path)
            temp_out = nib.Nifti1Image(x[0][3], affine=temp_in.affine)
            nib.save(temp_out, temp_out_path4)


            temp_out = nib.Nifti1Image(y[0][0], affine=temp_in.affine)
            nib.save(temp_out, temp_out_truth_path)

            # temp_out = nib.Nifti1Image(y[0][1], affine=temp_in.affine)
            # nib.save(temp_out, temp_out_truth_path2)

            # temp_out = nib.Nifti1Image(y[0][2], affine=temp_in.affine)
            # nib.save(temp_out, temp_out_truth_path3)

    print(n_validation_samples)


def test_patch():
    patch_overlap = [0, 0, 2]
    patch_overlap = np.asarray(patch_overlap)
    index_list = compute_patch_indices(image_shape=(160, 192, 128),
                                       patch_size=(160, 192, 7),
                                       overlap=patch_overlap, start=None,
                                       is_extract_patch_agressive=True,
                                       is_predict=False)

    print(index_list)
    return index_list


if __name__ == "__main__":
    # main25d(False)
    test_patch()
