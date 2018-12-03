import glob
import os
import shutil
import ntpath
from unet3d.utils.print_utils import print_processing, print_section
from brats.preprocess import get_image

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
if config["mode"] == "TEST":
    config["dataset"] = ["test"]
else:
    config["dataset"] = ["original", "preprocessed",
                                  "denoised_original", "denoised_preprocessed"]


def find_truth_filename(subject_dir, denoised_folder):
    seg_path = subject_dir.replace(denoised_folder, 'original_bak')
    file_card = os.path.join(seg_path, "*" + "seg" + ".nii.gz")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find groundtruth for", subject_dir)


def delete_previous_groundtruth(subject_dir):
    for truth in config["groundtruth_modalities"]:
        file_card = os.path.join(subject_dir, "*" + truth + ".nii.gz")
        for groundtruth_file in glob.glob(file_card):
            print("removing", groundtruth_file)
            os.remove(groundtruth_file)


def copy_groundtruth(subject_dir, truth_path):
    out_file = subject_dir + "/truth" + ".nii.gz"
    print("copying {} to {}".format(truth_path, out_file))
    shutil.copy(truth_path, out_file)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def rename_file(subject_file, modality):
    filename = path_leaf(subject_file)
    out_file = subject_file.replace(filename, modality) + ".nii.gz"
    print("renaming {} to {}".format(subject_file, out_file))
    os.rename(subject_file, out_file)


def main(mode="TEST"):
    for data_folder in config["data_folders"]:
        for denoised_folder in config["dataset"]:
            if config["env"] != "SERVER":
                subject_dirs = glob.glob(os.path.join(os.path.dirname(__file__), data_folder, denoised_folder, "*", "*"))
            else:
                subject_dirs = glob.glob(os.path.join(os.getcwd(), data_folder, denoised_folder, "*", "*"))
            for subject_dir in subject_dirs:
                if data_folder != "data_valid":
                    delete_previous_groundtruth(subject_dir)
                    truth_path = find_truth_filename(subject_dir, denoised_folder)
                    copy_groundtruth(subject_dir, truth_path)
                for modality in config["all_modalities"]:
                    subject_file = get_image(subject_dir, modality)
                    rename_file(subject_file, modality)


if __name__ == "__main__":
    print("------------------------------------------------------------------------------------")
    print(os.path.dirname(__file__))
    print(os.getcwd())
    main(config["mode"])