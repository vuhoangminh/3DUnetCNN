import random
from unet3d.utils.path_utils import get_project_dir
from brats.config import config, config_unet, config_dict
import datetime
import logging
import threading
import subprocess
import os
import sys
from subprocess import Popen, PIPE, STDOUT

from unet3d.utils.path_utils import make_dir
from unet3d.utils.path_utils import get_model_h5_filename
from unet3d.utils.path_utils import get_filename_without_extension

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


task = "finetune"
is_test = "0"

model_list = list()
cmd_list = list()
out_file_list = list()

for model_name in ["unet", "isensee"]:
    for is_denoise in config_dict["is_denoise"]:
        for is_normalize in config_dict["is_normalize"]:
            for is_hist_match in ["0", "1"]:
                for loss in ["weighted"]:
                    for patch_shape in ["160-192-7"]:
                        model_dim = 25

                        model_filename = get_model_h5_filename(
                            datatype="model",
                            is_bias_correction="1",
                            is_denoise=is_denoise,
                            is_normalize=is_normalize,
                            is_hist_match=is_hist_match,
                            depth_unet=4,
                            n_base_filters_unet=16,
                            model_name=model_name,
                            patch_shape=patch_shape,
                            is_crf="0",
                            is_test=is_test,
                            loss=loss,
                            model_dim=model_dim)

                        cmd = "python brats/{}.py -t \"{}\" -o \"0\" -n \"{}\" -de \"{}\" -hi \"{}\" -ps \"{}\" -l \"{}\" -m \"{}\" -ba 64 -dim 25".format(
                            task,
                            is_test,
                            is_normalize,
                            is_denoise,
                            is_hist_match,
                            patch_shape,
                            loss,
                            model_name
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
