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

# pp = pprint.PrettyPrinter(indent=4)
# # pp.pprint(config)
config.update(config_unet)


def run(model_filename, out_file, cmd):

    print("="*120)
    print(">> processing:", cmd)
    print("log to:", out_file)
    print(cmd)

    # out_file = open(out_file, 'w')

    # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
    #                         stderr=subprocess.STDOUT, shell=True)

    # # for line in proc.stdout:
    # #     line = line.decode("utf-8")
    # #     sys.stdout.write(line)
    # #     out_file.write(line)

    # proc.wait()

    os.system(cmd)

    # out_file.close()


task = "finetune"
is_test = "0"

model_list = list()
cmd_list = list()
out_file_list = list()

for model_name in ["unet", "seunet", "isensee"]:
    for is_denoise in config_dict["is_denoise"]:
        for is_normalize in config_dict["is_normalize"]:
            for is_hist_match in ["0", "1"]:
                patch_shape = "160-192-1"
                # patch_shape = "128-128-128"
                loss = "minh"

                log_folder = "log"
                make_dir(log_folder)

                d = datetime.date.today()
                year_current = d.year
                month_current = '{:02d}'.format(d.month)
                date_current = '{:02d}'.format(d.day)

                model_filename = get_filename_without_extension(get_model_h5_filename(
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
                    model_dim=2))

                out_file = "{}/{}{}{}_{}_out.log".format(
                    log_folder,
                    year_current,
                    month_current,
                    date_current,
                    model_filename)

                cmd = "python brats/{}.py -t \"{}\" -o \"0\" -n \"{}\" -de \"{}\" -hi \"{}\" -ps \"{}\" -l \"{}\" -m \"{}\" -ba 64 -dim 2".format(
                    task,
                    is_test,
                    is_normalize,
                    is_denoise,
                    is_hist_match,
                    patch_shape,
                    "minh",
                    model_name
                )

                model_list.append(model_filename)
                out_file_list.append(out_file)
                cmd_list.append(cmd)


import random
combined = list(zip(model_list, out_file_list, cmd_list))
random.shuffle(combined)

model_list[:], out_file_list[:], cmd_list = zip(*combined)

for i in range(len(model_list)):
    model_filename = model_list[i]
    out_file = out_file_list[i]
    cmd = cmd_list[i]

    # print("*"*60)
    # print(">> processing:")
    # print(model_filename)
    # print(out_file)
    # print(cmd)
    # print("*"*60)

    run(model_filename, out_file, cmd)