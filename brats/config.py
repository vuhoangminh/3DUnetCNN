config = dict()
config["env"] = "SERVER"  # change this to "FULL" if you want to run full
# config["mode"] = "TEST"  # change this to "FULL" if you want to run full
config["mode"] = "FULL"  # change this to "FULL" if you want to run full
config["data_folders"] = ["data_train", "data_valid"]
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
# change this if you want to only use some of the modalities
config["training_modalities"] = config["all_modalities"]
config["truth_old"] = ["seg"]
config["truth"] = ["truth"]
config["groundtruth_modalities"] = config["truth_old"] + config["truth"]
config["mask"] = "mask"
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
