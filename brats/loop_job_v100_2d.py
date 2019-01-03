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

task = "finetune"
is_test = "0"

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

                print("="*120)
                print(">> processing:", is_denoise,
                      is_normalize, is_hist_match, model_name)
                print("log to:", out_file)
                print(cmd)

                out_file = open(out_file, 'w')

                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, shell=True)
                for line in proc.stdout:
                    line = line.decode("utf-8")
                    sys.stdout.write(line)
                    out_file.write(line)

                proc.wait()
                out_file.close()
