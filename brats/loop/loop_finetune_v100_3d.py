import unet3d.utils.args_utils as get_args
from unet3d.utils.path_utils import get_model_h5_filename
import random
from unet3d.utils.path_utils import get_project_dir
from brats.config import config, config_unet
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

config.update(config_unet)

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def run(model_filename, cmd):

    print("="*120)

    model_path = os.path.join(
        BRATS_DIR, "database/model/finetune", model_filename)
    if os.path.exists(model_path):
        print("{} exists. Will skip!!".format(model_path))
    else:
        print(">> RUNNING:", cmd)
        from keras import backend as K
        os.system(cmd)
        K.clear_session()


args = get_args.train()
task = "finetune"
args.is_test = "0"

model_list = list()
cmd_list = list()
out_file_list = list()

for is_augment in ["0"]:
    args.is_augment = is_augment
    for model_name in ["isensee"]:
        args.model = model_name
        for is_denoise in ["0", "median", "bm4d"]:
            args.is_denoise = is_denoise
            for is_normalize in ["z", "01"]:
                args.is_normalize = is_normalize
                for is_hist_match in ["0", "1"]:
                    args.is_hist_match = is_hist_match
                    for loss in ["weighted", "minh"]:
                        for patch_shape in ["160-192-128"]:
                            args.patch_shape = patch_shape
                            model_dim = 3

                            if is_normalize == "z" and is_hist_match == "1":
                                continue

                            model_filename = get_model_h5_filename(
                                "model", args)

                            cmd = "python brats/{}.py -t \"{}\" -o \"0\" -n \"{}\" -de \"{}\" -hi \"{}\" -ps \"{}\" -l \"{}\" -m \"{}\" -ba 1 -au {} -dim 3".format(
                                task,
                                args.is_test,
                                args.is_normalize,
                                args.is_denoise,
                                args.is_hist_match,
                                args.patch_shape,
                                args.loss,
                                args.model,
                                args.is_augment
                            )

                            model_list.append(model_filename)
                            cmd_list.append(cmd)


combined = list(zip(model_list, cmd_list))
random.shuffle(combined)

model_list[:], cmd_list = zip(*combined)

for i in range(len(model_list)):
    model_filename = model_list[i]
    cmd = cmd_list[i]
    run(model_filename, cmd)
