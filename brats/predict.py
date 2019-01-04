import os

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


def predict(overwrite=True, crop=True, challenge="brats", year=2018,
            image_shape="160-192-128", is_bias_correction="1",
            is_normalize="z", is_denoise="0",
            is_hist_match="0", is_test="1",
            depth_unet=4, n_base_filters_unet=16, model_name="unet",
            patch_shape="128-128-128", is_crf="0",
            batch_size=1, loss="minh", model_dim=3):

    data_path, trainids_path, validids_path, testids_path, model_path = get_training_h5_paths(
        brats_dir=BRATS_DIR, overwrite=overwrite, crop=crop, challenge=challenge, year=year,
        image_shape=image_shape, is_bias_correction=is_bias_correction,
        is_normalize=is_normalize, is_denoise=is_denoise,
        is_hist_match=is_hist_match, is_test=is_test,
        model_name=model_name, depth_unet=depth_unet, n_base_filters_unet=n_base_filters_unet,
        patch_shape=patch_shape, is_crf=is_crf, loss=loss, model_dim=model_dim,
        dir_read_write="finetune", is_finetune=True)

    config["data_file"] = data_path
    config["model_file"] = model_path
    config["training_file"] = trainids_path
    config["validation_file"] = validids_path
    config["testing_file"] = testids_path
    config["patch_shape"] = get_shape_from_string(patch_shape)
    config["input_shape"] = tuple(
        [config["nb_channels"]] + list(config["patch_shape"]))
    config["prediction_folder"] = os.path.join(
        BRATS_DIR, "database/prediction", get_filename_without_extension(config["model_file"]))
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

    if model_dim == 3:
        from unet3d.prediction import run_validation_cases
    elif model_dim == 25:
        from unet25d.prediction import run_validation_cases
    elif model_dim == 2:
        from unet2d.prediction import run_validation_cases
    else:
        raise ValueError("dim {} NotImplemented error. Please check".format(model_dim))

    run_validation_cases(validation_keys_file=config["testing_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=config["prediction_folder"])


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
    model_dim = args.model_dim

    # for is_normalize in config_dict["is_normalize"]:
    #     for is_denoise in config_dict["is_denoise"]:
    #         for is_hist_match in ["0", "1"]:
    
    
    # # ================3d==================================
    # for is_normalize in ["z"]:
    #     for is_denoise in ["0"]:
    #         for is_hist_match in ["0"]:
    #             for model_name in ["unet", "isensee"]:
    #                 # patch_shape = "160-192-128"
    # # ================3d==================================  
      
    # # ================2d==================================             
    # model_dim = 2
    # batch_size = 64
    # for is_normalize in ["z"]:
    #     for is_denoise in ["0"]:
    #         for is_hist_match in ["0"]:
    #             for model_name in ["unet", "seunet"]:
    #                 patch_shape = "160-192-1"   
    # # ================2d==================================             

    # ================25d==================================             
    model_dim = 2
    batch_size = 64
    for is_normalize in ["z"]:
        for is_denoise in ["0"]:
            for is_hist_match in ["0"]:
                for model_name in ["unet", "seunet"]:
                    patch_shape = "160-192-17"                          
    # ================25d==================================                                             
                    loss = "minh"
                    print("="*60)
                    print(">> processing:", is_denoise,
                          is_normalize, is_hist_match, model_name)
                    is_test = "0"
                    predict(overwrite=overwrite, crop=crop, challenge=challenge, year=year,
                            image_shape=image_shape, is_bias_correction=is_bias_correction,
                            is_normalize=is_normalize, is_denoise=is_denoise,
                            is_hist_match=is_hist_match, is_test=is_test,
                            model_name=model_name, depth_unet=depth_unet, n_base_filters_unet=n_base_filters_unet,
                            patch_shape=patch_shape, is_crf=is_crf, batch_size=batch_size,
                            loss=loss, model_dim=model_dim)
                    print("="*60)
                    print(">> finished:", is_denoise,
                          is_normalize, is_hist_match, model_name)
                    print("="*60)


if __name__ == "__main__":
    main()
