config = dict()
config["env"] = "SERVER"  # change this to "FULL" if you want to run full
# config["mode"] = "TEST"  # change this to "FULL" if you want to run full
config["mode"] = "FULL"  # change this to "FULL" if you want to run full
config["data_folders"] = ["data_train", "data_valid"]
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"] # change this if you want to only use some of the modalities
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
config["brats_folder"] = "brats"
config["dataset_folder"] = "dataset"
config["template_data_folder"] = "data_train"
config["template_folder"] = "HGG/Brats18_2013_2_1"
# config["is_create_patch_index_list_original"] = False


config["labels"] = (1, 2, 4)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])



# configs of u-net
config_unet = dict()
config_unet["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config_unet["image_shape"] = (240, 240, 155)  # This determines what shape the images will be cropped/resampled to.
config_unet["patch_shape"] = (128, 128, 128)  # switch to None to train on the whole image

if "patch_shape" in config_unet and config_unet["patch_shape"] is not None:
    config_unet["input_shape"] = tuple([config["nb_channels"]] + list(config_unet["patch_shape"]))
else:
    config_unet["input_shape"] = tuple([config["nb_channels"]] + list(config_unet["image_shape"]))

config_unet["truth_channel"] = config["nb_channels"]
config_unet["deconvolution"] = True  # if False, will use upsampling instead of deconvolution
config_unet["depth"] = 4
config_unet["n_base_filters"] = 16

config_unet["batch_size"] = 1
config_unet["validation_batch_size"] = 2
config_unet["n_epochs"] = 500  # cutoff the training after this many epochs
config_unet["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config_unet["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config_unet["initial_learning_rate"] = 0.00001
config_unet["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config_unet["validation_split"] = 0.8  # portion of the data that will be used for training
config_unet["flip"] = False  # augments the data by randomly flipping an axis during
config_unet["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config_unet["distort"] = None  # switch to None if you want no distortion
config_unet["augment"] = config_unet["flip"] or config_unet["distort"]
config_unet["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config_unet["training_patch_start_offset"] = None # randomly offset the first patch index by up to this offset
config_unet["skip_blank"] = True  # if True, then patches without any target will be skipped