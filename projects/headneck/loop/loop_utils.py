from unet3d.utils.path_utils import get_project_dir
import os


def run(model_filename, cmd, config, model_path="database/model/finetune", mode_run=2):

    # mode_run: 0 - just run
    # mode_run: 1 - run if file not exists
    # mode_run: 2 - run if file not exists and no gpu is running

    CURRENT_WORKING_DIR = os.path.realpath(__file__)
    PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
    BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
    DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])

    def check_is_running(model_filename):
        path = os.path.join(BRATS_DIR, "loop/running.txt")
        is_running = False
        if not os.path.exists(path):
            f = open(path, "w")
            f.write("{}\n".format(model_filename))
            f.close()
        else:
            f = open(path, "r")
            lines = f.readlines()
            f.close()
            for line in lines:
                if model_filename in line:
                    is_running = True
            if not is_running:
                f = open(path, "a+")
                f.write("{}\n".format(model_filename))
                f.close()
        return is_running

    print("="*120)
    model_path = os.path.join(BRATS_DIR, model_path, model_filename)

    is_run = False
    if mode_run == 0:
        is_run = True
    if mode_run == 1 and not os.path.exists(model_path):
        is_run = True
    if mode_run == 2 and not os.path.exists(model_path) and not check_is_running(model_filename):
        is_run = True

    if is_run:
        try:
            print(">> RUNNING:", cmd)
            from keras import backend as K
            os.system(cmd)
            K.clear_session()
        except:
            print("something wrong")
    else:
        print("{} exists or in training. Will skip!!".format(model_path))
