config = dict()
config["env"] = "SERVER"  # change this to "FULL" if you want to run full
# config["mode"] = "TEST"  # change this to "FULL" if you want to run full
config["mode"] = "FULL"  # change this to "FULL" if you want to run full
config["data_folders"] = ["data_train", "data_valid"]
# change this if you want to only use some of the modalities
config["all_modalities"] = ["t1"]
config["training_modalities"] = config["all_modalities"]
config["nb_channels"] = len(config["training_modalities"])
config["truth_old"] = ["seg"]
config["truth"] = ["truth"]
config["groundtruth_modalities"] = config["truth_old"] + config["truth"]
config["mask"] = ["mask"]
if config["mode"] == "TEST":
    config["dataset"] = ["test"]
else:
    config["dataset"] = ["original", "preprocessed",
                         "denoised_original", "denoised_preprocessed",
                         "test"]
config["dataset_minh_normalize"] = ["original_minh_normalize", "preprocessed_minh_normalize",
                                    "denoised_original_minh_normalize", "denoised_preprocessed_minh_normalize",
                                    "test_minh_normalize"]
config["original_folder"] = ["original_bak"]
config["project_name"] = "3DUnetCNN_BRATS"
config["brats_folder"] = "projects/ibsr"
config["dataset_folder"] = "projects/ibsr/database"
config["template_data_folder"] = "database/data_train"
config["template_folder"] = "IBSR_01"

# config_unet["image_shape"] = (240, 240, 155)  # This determines what shape the images will be cropped/resampled to.
# This determines what shape the images will be cropped/resampled to.
# config["image_shape"] = (160, 192, 128)
config["image_shape"] = (144,144,144)
# config["is_create_patch_index_list_original"] = False


config["labels"] = (1, 2, 3)  # the label numbers on the input image
# config["labels"] = (0, 1, 2, 4)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])


# configs of u-net
config_unet = dict()
# pool size for the max pooling operations
config_unet["pool_size"] = (2, 2, 2)
# switch to None to train on the whole image
config_unet["patch_shape"] = (128, 128, 128)

if "patch_shape" in config_unet and config_unet["patch_shape"] is not None:
    config_unet["input_shape"] = tuple(
        [config["nb_channels"]] + list(config_unet["patch_shape"]))
else:
    config_unet["input_shape"] = tuple(
        [config["nb_channels"]] + list(config_unet["image_shape"]))

config_unet["truth_channel"] = config["nb_channels"]
# if False, will use upsampling instead of deconvolution
config_unet["deconvolution"] = True
config_unet["depth"] = 4
config_unet["n_base_filters"] = 16

config_unet["batch_size"] = 1
config_unet["validation_batch_size"] = 2
config_unet["n_epochs"] = 200  # cutoff the training after this many epochs
# learning rate will be reduced after this many epochs if the validation loss is not improving
# config_unet["patience"] = 10
config_unet["patience"] = 20
# training will be stopped after this many epochs without the validation loss improving
config_unet["early_stop"] = 50
config_unet["initial_learning_rate"] = 1e-4 # factor by which the learning rate will be reduced
config_unet["learning_rate_drop"] = 0.2 # portion of the data that will be used for training
# config_unet["learning_rate_epochs"] = 1
config_unet["validation_split"] = 0.8 # if > 0, during training, validation patches will be overlapping
config_unet["validation_patch_overlap"] = 0 # randomly offset the first patch index by up to this offset
config_unet["training_patch_start_offset"] = None

# if False, extract patches only in bouding box of mask
config_unet["is_create_patch_index_list_original"] = True


config["augment_flipud"] = False
# config["augment_fliplr"] = True
config["augment_fliplr"] = False
# config["augment_elastic"] = True
config["augment_elastic"] = False
config["augment_rotation"] = False
# config["augment_rotation"] = True
config["augment_shift"] = False
config["augment_shear"] = False
config["augment_zoom"] = False
config["n_augment"] = 0

config["flip"] = False  # augments the data by randomly flipping an axis during
# data shape must be a cube. Augments the data by permuting in various directions
config["permute"] = True
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
# if True, then patches without any target will be skipped
config["skip_blank"] = True


# Dictionary
config_dict = dict()
config_dict["challenge"] = ["brats"]
config_dict["year"] = [2018, 2019]
config_dict["model"] = ["unet", "isensee", "densefcn", "denseunet", "resunet", "seunet", "seisensee", "simple", "eye", "m", "m2", "multi"]
config_dict["model_depth"] = ["unet", "seunet", "multi", "denseunet", "resunet"]
# "deepmedic", "maskrcnn", "cascaded", "proposed"]
config_dict["depth_unet"] = [3, 4, 5, 6]  # depth of unet
config_dict["n_base_filters_unet"] = [4, 8, 16, 32]  # number of base filters of unet
config_dict["image_shape"] = ["160-192-128", "144-144-144", "240-240-155"]
config_dict["patch_shape"] = ["16-16-16", "32-32-32", "64-64-64", "128-128-128", "160-192-128", "160-192-1", "160-192-7"]
config_dict["is_bias_correction"] = ["0","1"]
config_dict["is_denoise"] = ["0", "bm4d", "gaussian", "median"]
config_dict["is_normalize"] = ["z", "01"]
config_dict["is_crf"] = ["0", "post", "cnn", "rnn"]
config_dict["crop"] = ["0", "1"]
config_dict["hist_match"] = ["0", "1"]
config_dict["loss"] = ["weighted", "minh", "tversky", "tv_minh"]


config_convert_name = {
    "original": "bias-0_denoise-0",
    "preprocessed": "bias-1_denoise-0",
    "denoised_original": "bias-0_denoise-bm4d",
    "denoised_preprocessed": "bias-1_denoise-bm4d",    
}

config_finetune = dict()
config_finetune["n_epochs"] = 100  # cutoff the training after this many epochs
# learning rate will be reduced after this many epochs if the validation loss is not improving
config_finetune["patience"] = 5
# training will be stopped after this many epochs without the validation loss improving
config_finetune["early_stop"] = 16
config_finetune["initial_learning_rate"] = 4e-5 # factor by which the learning rate will be reduced
config_finetune["learning_rate_drop"] = 0.2 # portion of the data that will be used for training