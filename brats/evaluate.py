from brats.config import config, config_unet, config_dict
from unet3d.utils.path_utils import make_dir
from unet3d.utils.path_utils import get_filename_without_extension
from unet3d.utils.path_utils import get_shape_from_string
from unet3d.utils.path_utils import get_training_h5_paths
from unet3d.utils.path_utils import get_project_dir
import unet3d.utils.args_utils as get_args
from unet3d.prediction import run_validation_cases
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
from medpy.metric.binary import hd, sensitivity, specificity, dc

matplotlib.use('agg')
voxelspacings = [(0.9375, 1.5, 0.9375), (1, 1.5, 1), (0.8371, 1.5, 0.8371)]

config.update(config_unet)

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, config["project_name"])
BRATS_DIR = os.path.join(PROJECT_DIR, config["brats_folder"])
DATASET_DIR = os.path.join(PROJECT_DIR, config["dataset_folder"])


def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def get_score(truth, prediction, masking_functions):
    score = list()
    dice_score = [dc(func(truth), func(prediction))
                  for func in masking_functions]
    # hd_score = [hd(func(truth), func(prediction))
    #             for func in masking_functions]
    sensitivity_score = [sensitivity(func(truth), func(prediction))
                         for func in masking_functions]
    specificity_score = [specificity(func(truth), func(prediction))
                         for func in masking_functions]
    score.extend(dice_score)
    # score.extend(sensitivity_score)
    # score.extend(specificity_score)
    return score
    # extend(hd_score)


def get_model_info_header(challenge, year, image_shape, is_bias_correction,
                          is_denoise, is_normalize, is_hist_match,
                          model_name, model_dim, patch_shape, loss):
    return [challenge, str(year), image_shape, is_bias_correction, is_denoise,
            is_normalize, is_hist_match, model_name, str(model_dim), loss, patch_shape]


def evaluate(overwrite=True, crop=True, challenge="brats", year=2018,
             image_shape="160-192-128", is_bias_correction="1",
             is_normalize="z", is_denoise="0",
             is_hist_match="0", is_test="1",
             depth_unet=4, n_base_filters_unet=16, model_name="unet",
             patch_shape="128-128-128", is_crf="0",
             batch_size=1, loss="weighted", model_dim=3):

    header = ("dice_WholeTumor", "dice_TumorCore", "dice_EnhancingTumor")

    masking_functions = (get_whole_tumor_mask,
                         get_tumor_core_mask,
                         get_enhancing_tumor_mask)

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
    
    prediction_dir = "/mnt/sda/3DUnetCNN_BRATS/brats"
        # prediction_dir = BRATS_DIR
    config["prediction_folder"] = os.path.join(
        prediction_dir, "database/prediction", get_filename_without_extension(config["model_file"]))

    if not os.path.exists(config["prediction_folder"]):
        print("model not exists. Please check")
        return None, None
    else:
        prediction_df_csv_folder = os.path.join(
            BRATS_DIR, "database/prediction/csv/")
        make_dir(prediction_df_csv_folder)

        config["prediction_df_csv"] = prediction_df_csv_folder + \
            get_filename_without_extension(config["model_file"]) + ".csv"

        print("-"*60)
        print("SUMMARY")
        print("-"*60)
        print("model file:", config["model_file"])
        print("prediction folder:", config["prediction_folder"])
        print("csv file:", config["prediction_df_csv"])
        print("-"*60)

        rows = list()
        subject_ids = list()
        for case_folder in glob.glob(os.path.join(config["prediction_folder"], "*")):
            if not os.path.isdir(case_folder):
                continue
            subject_ids.append(os.path.basename(case_folder))
            truth_file = os.path.join(case_folder, "truth.nii.gz")
            truth_image = nib.load(truth_file)
            truth = truth_image.get_data()
            prediction_file = os.path.join(case_folder, "prediction.nii.gz")
            prediction_image = nib.load(prediction_file)
            prediction = prediction_image.get_data()
            score_case = get_score(truth, prediction, masking_functions)
            rows.append(score_case)

        df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
        df.to_csv(config["prediction_df_csv"])

        scores = dict()
        for index, score in enumerate(df.columns):
            values = df.values.T[index]
            scores[score] = values[np.isnan(values) == False]

        return scores, model_path
    # plot_prediction(df)


def plot_prediction(df):
    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("validation_scores_boxplot.png")
    plt.close()

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph.png')


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

    header = ("dice_WholeTumor", "dice_TumorCore", "dice_EnhancingTumor")
    model_scores = list()
    model_ids = list()

    # for depth_unet in [3,4]:
    #     for n_base_filters_unet in [16,32]:
    for model_dim in [2, 3, 25]:
        for model_name in config_dict["model"]:
            for is_normalize in config_dict["is_normalize"]:
                for is_denoise in config_dict["is_denoise"]:
                    for is_hist_match in config_dict["hist_match"]:
                        for patch_shape in config_dict["patch_shape"]:
                            for loss in config_dict["loss"]:
                                print("="*120)
                                print(
                                    ">> processing model-{}{}, patch_shape-{}, is_denoise-{}, is_normalize-{}, is_hist_match-{}, loss-{}".format(
                                        model_name,
                                        model_dim,
                                        patch_shape,
                                        is_denoise,
                                        is_normalize,
                                        is_hist_match,
                                        loss))
                                is_test = "0"
                                print("="*120)
                                print(">> processing:", is_denoise,
                                    is_normalize, is_hist_match, model_name)
                                is_test = "0"
                                model_score, model_path = evaluate(overwrite=overwrite, crop=crop, challenge=challenge, year=year,
                                                                image_shape=image_shape, is_bias_correction=is_bias_correction,
                                                                is_normalize=is_normalize, is_denoise=is_denoise,
                                                                is_hist_match=is_hist_match, is_test=is_test,
                                                                model_name=model_name, depth_unet=depth_unet,
                                                                n_base_filters_unet=n_base_filters_unet,
                                                                patch_shape=patch_shape, is_crf=is_crf, batch_size=batch_size,
                                                                loss=loss, model_dim=model_dim)
                                if model_score is not None:
                                    print("="*120)
                                    print(">> finished:")

                                    model_ids.append(
                                        get_filename_without_extension(model_path))

                                    row = get_model_info_header(challenge, year, image_shape, is_bias_correction,
                                                                is_denoise, is_normalize, is_hist_match,
                                                                model_name, model_dim, patch_shape, loss)
                                    score = [np.mean(model_score["dice_WholeTumor"]),
                                            np.mean(
                                                model_score["dice_TumorCore"]),
                                            np.mean(
                                                model_score["dice_EnhancingTumor"]),
                                            (np.mean(model_score["dice_WholeTumor"])+np.mean(model_score["dice_TumorCore"])+np.mean(model_score["dice_EnhancingTumor"]))/3]

                                    row.extend(score)
                                    model_scores.append(row)

    header = ("challenge", "year", "image_shape", "is_bias_correction",
              "is_denoise", "is_normalize", "is_hist_match",
              "model_name", "model_dim", "loss", "patch_shape", 
              "dice_WholeTumor", "dice_TumorCore",
              "dice_EnhancingTumor", "dice_Mean")
              
    final_df = pd.DataFrame.from_records(
        model_scores, columns=header, index=model_ids)

    print(final_df)

    prediction_df_csv_folder = os.path.join(
        BRATS_DIR, "database/prediction/csv/")
    make_dir(prediction_df_csv_folder)

    to_file = prediction_df_csv_folder + "compile.csv"

    final_df.to_csv(to_file)


if __name__ == "__main__":
    main()
