import argparse
from .utils import str2bool
from brats.config import config_dict


def parent_train_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-o', '--overwrite', type=str2bool,
                        default="False")
    parser.add_argument('-n', '--is_normalize', type=str,
                        default="z", choices=config_dict["is_normalize"],
                        help="what type of normalization")
    parser.add_argument('-de', '--is_denoise', type=str,
                        default="0", choices=config_dict["is_denoise"],
                        help="what type of denoising")
    parser.add_argument('-t', '--is_test', type=str,
                        default="0", choices=["0", "1"])
    parser.add_argument('-hi', '--is_hist_match', type=str,
                        default="0")
    parser.add_argument('-m', '--model', type=str,
                        default="unet", choices=config_dict["model"])
    parser.add_argument('-du', '--depth_unet', type=int,
                        default=4, choices=config_dict["depth_unet"])
    parser.add_argument('-nb', '--n_base_filters_unet', type=int,
                        default=16,
                        )
    parser.add_argument('-crf', '--is_crf', type=str,
                        default="0", choices=config_dict["is_crf"],
                        help="crf method")
    parser.add_argument('-l', '--loss', type=str,
                        default="weighted",
                        help="loss function")
    parser.add_argument('-alpha_tv', '--weight_tv_to_main_loss', type=float,
                        default=0.1,
                        help="weight of total variance comapared to main loss (dice coef etc.)")
    parser.add_argument('-au', '--is_augment', type=str,
                        default="1")                        
    return parser


def parent_prepare_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-o', '--overwrite', type=str2bool,
                        default="False")
    parser.add_argument('-n', '--is_normalize', type=str,
                        default="z", choices=config_dict["is_normalize"],
                        help="what type of normalization")
    parser.add_argument('-de', '--is_denoise', type=str,
                        default="0", choices=config_dict["is_denoise"],
                        help="what type of denoising")
    parser.add_argument('-t', '--is_test', type=str,
                        default="1", choices=["0", "1"])
    parser.add_argument('-hi', '--is_hist_match', type=str,
                        default="0")
    return parser


def finetune():
    parent_parser = parent_train_parser()
    parser = argparse.ArgumentParser(
        parents=[parent_parser], description='Finetuning')
    parser.add_argument('-c', '--challenge', type=str,
                        default="brats", choices=config_dict["challenge"],
                        help="challenge name")
    parser.add_argument('-y', '--year', type=int,
                        default=2018, choices=config_dict["year"],
                        help="year of challenge")
    parser.add_argument('-is', '--image_shape', type=str,
                        default="160-192-128", choices=config_dict["image_shape"],
                        help="image shape to read")
    parser.add_argument('-ps', '--patch_shape', type=str,
                        default="128-128-128",
                        help="patch shape to train")
    parser.add_argument('-ba', '--batch_size', type=int,
                        default=1,
                        help="train batch size")
    parser.add_argument('-dim', '--model_dim', type=int,
                        default=3,
                        help="2d, 2.5d or 3d?")
    parser.add_argument('-r', '--crop', type=str,
                        default="1", choices=config_dict["crop"])
    parser.add_argument('-b', '--is_bias_correction', type=str,
                        default="1", choices=config_dict["is_bias_correction"],
                        help="perform bias field removal?")
    args = parser.parse_args()
    return args


def train():
    parent_parser = parent_train_parser()
    parser = argparse.ArgumentParser(
        parents=[parent_parser], description='Finetuning')
    parser.add_argument('-c', '--challenge', type=str,
                        default="brats", choices=config_dict["challenge"],
                        help="challenge name")
    parser.add_argument('-y', '--year', type=int,
                        default=2018, choices=config_dict["year"],
                        help="year of challenge")
    parser.add_argument('-is', '--image_shape', type=str,
                        default="160-192-128", choices=config_dict["image_shape"],
                        help="image shape to read")
    parser.add_argument('-ps', '--patch_shape', type=str,
                        default="128-128-128",
                        help="patch shape to train")
    parser.add_argument('-ba', '--batch_size', type=int,
                        default=1,
                        help="train batch size")
    parser.add_argument('-r', '--crop', type=str,
                        default="1", choices=config_dict["crop"])
    parser.add_argument('-b', '--is_bias_correction', type=str,
                        default="1", choices=config_dict["is_bias_correction"],
                        help="perform bias field removal?")
    parser.add_argument('-dim', '--model_dim', type=int,
                        default=3)                        
    args = parser.parse_args()
    return args


def train2d():
    parent_parser = parent_train_parser()
    parser = argparse.ArgumentParser(
        parents=[parent_parser], description='Finetuning')
    parser.add_argument('-c', '--challenge', type=str,
                        default="brats", choices=config_dict["challenge"],
                        help="challenge name")
    parser.add_argument('-y', '--year', type=int,
                        default=2018, choices=config_dict["year"],
                        help="year of challenge")
    parser.add_argument('-is', '--image_shape', type=str,
                        default="160-192-128", choices=config_dict["image_shape"],
                        help="image shape to read")
    parser.add_argument('-ps', '--patch_shape', type=str,
                        default="160-192-1",
                        help="patch shape to train")
    parser.add_argument('-ba', '--batch_size', type=int,
                        default=64,
                        help="train batch size")
    parser.add_argument('-r', '--crop', type=str,
                        default="1", choices=config_dict["crop"])
    parser.add_argument('-b', '--is_bias_correction', type=str,
                        default="1", choices=config_dict["is_bias_correction"],
                        help="perform bias field removal?")
    parser.add_argument('-dim', '--model_dim', type=int,
                        default=2)                        
    args = parser.parse_args()
    return args


def train25d():
    parent_parser = parent_train_parser()
    parser = argparse.ArgumentParser(
        parents=[parent_parser], description='Finetuning')
    parser.add_argument('-c', '--challenge', type=str,
                        default="brats", choices=config_dict["challenge"],
                        help="challenge name")
    parser.add_argument('-y', '--year', type=int,
                        default=2018, choices=config_dict["year"],
                        help="year of challenge")
    parser.add_argument('-is', '--image_shape', type=str,
                        default="160-192-128", choices=config_dict["image_shape"],
                        help="image shape to read")
    parser.add_argument('-ps', '--patch_shape', type=str,
                        default="160-192-9",
                        help="patch shape to train")
    parser.add_argument('-ba', '--batch_size', type=int,
                        default=64,
                        help="train batch size")
    parser.add_argument('-r', '--crop', type=str,
                        default="1", choices=config_dict["crop"])
    parser.add_argument('-b', '--is_bias_correction', type=str,
                        default="1", choices=config_dict["is_bias_correction"],
                        help="perform bias field removal?")
    parser.add_argument('-dim', '--model_dim', type=int,
                        default=25)                        
    args = parser.parse_args()
    return args


def prepare_data():
    parent_parser = parent_prepare_parser()
    parser = argparse.ArgumentParser(
        parents=[parent_parser], description='Data preparation')
    parser.add_argument('-c', '--challenge', type=str,
                        default="brats", choices=config_dict["challenge"],
                        help="challenge name")
    parser.add_argument('-y', '--year', type=str,
                        default=2018, choices=config_dict["year"],
                        help="year of challenge")
    parser.add_argument('-is', '--image_shape', type=str,
                        default="160-192-128", choices=config_dict["image_shape"],
                        help="image shape to read")
    parser.add_argument('-r', '--crop', type=str,
                        default="1")
    parser.add_argument('-b', '--is_bias_correction', type=str,
                        default="1", choices=config_dict["is_bias_correction"],
                        help="perform bias field removal?")
    args = parser.parse_args()
    return args


def prepare_data_ibsr():
    parent_parser = parent_prepare_parser()
    parser = argparse.ArgumentParser(
        parents=[parent_parser], description='Data preparation')
    parser.add_argument('-c', '--challenge', type=str,
                        default="ibsr",
                        help="challenge name")
    parser.add_argument('-y', '--year', type=str,
                        default=2018, choices=config_dict["year"],
                        help="year of challenge")
    parser.add_argument('-is', '--image_shape', type=str,
                        default="256-128-256", choices=config_dict["image_shape"],
                        help="image shape to read")
    parser.add_argument('-r', '--crop', type=str,
                        default="0")
    parser.add_argument('-b', '--is_bias_correction', type=str,
                        default="0", choices=config_dict["is_bias_correction"],
                        help="perform bias field removal?")
    args = parser.parse_args()
    return args


def train_ibsr():
    parent_parser = parent_train_parser()
    parser = argparse.ArgumentParser(
        parents=[parent_parser], description='Finetuning')
    parser.add_argument('-c', '--challenge', type=str,
                        default="ibsr", choices=config_dict["challenge"],
                        help="challenge name")
    parser.add_argument('-y', '--year', type=str,
                        default=2018, choices=config_dict["year"],
                        help="year of challenge")
    parser.add_argument('-is', '--image_shape', type=str,
                        default="256-128-256", choices=config_dict["image_shape"],
                        help="image shape to read")
    parser.add_argument('-ps', '--patch_shape', type=str,
                        default="128-128-128",
                        help="patch shape to train")
    parser.add_argument('-ba', '--batch_size', type=int,
                        default=1,
                        help="train batch size")
    parser.add_argument('-r', '--crop', type=str,
                        default="0", choices=config_dict["crop"])
    parser.add_argument('-b', '--is_bias_correction', type=str,
                        default="0", choices=config_dict["is_bias_correction"],
                        help="perform bias field removal?")
    parser.add_argument('-dim', '--model_dim', type=int,
                        default=3)                        
    args = parser.parse_args()
    return args


def train2d_ibsr():
    parent_parser = parent_train_parser()
    parser = argparse.ArgumentParser(
        parents=[parent_parser], description='Finetuning')
    parser.add_argument('-c', '--challenge', type=str,
                        default="ibsr", choices=config_dict["challenge"],
                        help="challenge name")
    parser.add_argument('-y', '--year', type=str,
                        default=2018, choices=config_dict["year"],
                        help="year of challenge")
    parser.add_argument('-is', '--image_shape', type=str,
                        default="256-128-256", choices=config_dict["image_shape"],
                        help="image shape to read")
    parser.add_argument('-ps', '--patch_shape', type=str,
                        default="256-128-1",
                        help="patch shape to train")
    parser.add_argument('-ba', '--batch_size', type=int,
                        default=64,
                        help="train batch size")
    parser.add_argument('-r', '--crop', type=str,
                        default="0", choices=config_dict["crop"])
    parser.add_argument('-b', '--is_bias_correction', type=str,
                        default="0", choices=config_dict["is_bias_correction"],
                        help="perform bias field removal?")
    parser.add_argument('-dim', '--model_dim', type=int,
                        default=2)                        
    args = parser.parse_args()
    return args


# def finetune():
#     parser = argparse.ArgumentParser(description='Finetuning')
#     parser.add_argument('-o', '--overwrite', type=str2bool,
#                         default="False")
#     parser.add_argument('-r', '--crop', type=str,
#                         default="1", choices=config_dict["crop"])
#     parser.add_argument('-c', '--challenge', type=str,
#                         default="brats", choices=config_dict["challenge"],
#                         help="challenge name")
#     parser.add_argument('-y', '--year', type=int,
#                         default=2018, choices=config_dict["year"],
#                         help="year of challenge")
#     parser.add_argument('-is', '--image_shape', type=str,
#                         default="160-192-128", choices=config_dict["image_shape"],
#                         help="image shape to read")
#     parser.add_argument('-b', '--is_bias_correction', type=str,
#                         default="1", choices=config_dict["is_bias_correction"],
#                         help="perform bias field removal?")
#     parser.add_argument('-n', '--is_normalize', type=str,
#                         default="z", choices=config_dict["is_normalize"],
#                         help="what type of normalization")
#     parser.add_argument('-de', '--is_denoise', type=str,
#                         default="0", choices=config_dict["is_denoise"],
#                         help="what type of denoising")
#     parser.add_argument('-t', '--is_test', type=str,
#                         default="1", choices=["0", "1"])
#     parser.add_argument('-hi', '--is_hist_match', type=str,
#                         default="0")
#     parser.add_argument('-m', '--model', type=str,
#                         default="isensee", choices=config_dict["model"])
#     parser.add_argument('-du', '--depth_unet', type=int,
#                         default=4, choices=config_dict["depth_unet"])
#     parser.add_argument('-nb', '--n_base_filters_unet', type=int,
#                         default=16, choices=config_dict["n_base_filters_unet"])
#     parser.add_argument('-ps', '--patch_shape', type=str,
#                         default="128-128-128",
#                         help="patch shape to train")
#     parser.add_argument('-crf', '--is_crf', type=str,
#                         default="0", choices=config_dict["is_crf"],
#                         help="crf method")
#     parser.add_argument('-ba', '--batch_size', type=int,
#                         default=1,
#                         help="train batch size")
#     parser.add_argument('-l', '--loss', type=str,
#                         default="weighted",
#                         help="loss function")
#     parser.add_argument('-dim', '--model_dim', type=int,
#                         default=3,
#                         help="2d, 2.5d or 3d?")
#     parser.add_argument('-alpha_tv', '--weight_tv_to_main_loss', type=float,
#                         default=0.1,
#                         help="weight of total variance comapared to main loss (dice coef etc.)")
#     args = parser.parse_args()
#     return args


# def train():
#     parser = argparse.ArgumentParser(description='Training')
#     parser.add_argument('-o', '--overwrite', type=str2bool,
#                         default="False")
#     parser.add_argument('-r', '--crop', type=str,
#                         default="1", choices=config_dict["crop"])
#     parser.add_argument('-c', '--challenge', type=str,
#                         default="brats", choices=config_dict["challenge"],
#                         help="challenge name")
#     parser.add_argument('-y', '--year', type=int,
#                         default=2018, choices=config_dict["year"],
#                         help="year of challenge")
#     parser.add_argument('-is', '--image_shape', type=str,
#                         default="160-192-128", choices=config_dict["image_shape"],
#                         help="image shape to read")
#     parser.add_argument('-b', '--is_bias_correction', type=str,
#                         default="1", choices=config_dict["is_bias_correction"],
#                         help="perform bias field removal?")
#     parser.add_argument('-n', '--is_normalize', type=str,
#                         default="z", choices=config_dict["is_normalize"],
#                         help="what type of normalization")
#     parser.add_argument('-de', '--is_denoise', type=str,
#                         default="0", choices=config_dict["is_denoise"],
#                         help="what type of denoising")
#     parser.add_argument('-t', '--is_test', type=str,
#                         default="1", choices=["0", "1"])
#     parser.add_argument('-hi', '--is_hist_match', type=str,
#                         default="0")
#     parser.add_argument('-m', '--model', type=str,
#                         default="isensee", choices=config_dict["model"])
#     parser.add_argument('-du', '--depth_unet', type=int,
#                         default=4, choices=config_dict["depth_unet"])
#     parser.add_argument('-nb', '--n_base_filters_unet', type=int,
#                         default=16,
#                         # choices=config_dict["n_base_filters_unet"]
#                         )
#     parser.add_argument('-ps', '--patch_shape', type=str,
#                         default="128-128-128", choices=config_dict["patch_shape"],
#                         help="patch shape to train")
#     parser.add_argument('-crf', '--is_crf', type=str,
#                         default="0", choices=config_dict["is_crf"],
#                         help="crf method")
#     parser.add_argument('-ba', '--batch_size', type=int,
#                         default=1,
#                         help="train batch size")
#     parser.add_argument('-l', '--loss', type=str,
#                         default="weighted",
#                         help="loss function")
#     parser.add_argument('-dim', '--model_dim', type=int,
#                         default=3,
#                         help="2d, 2.5d or 3d?")
#     parser.add_argument('-alpha_tv', '--weight_tv_to_main_loss', type=float,
#                         default=0.1,
#                         help="weight of total variance comapared to main loss (dice coef etc.)")
#     args = parser.parse_args()
#     return args


# def train2d():
#     parser = argparse.ArgumentParser(description='Training')
#     parser.add_argument('-o', '--overwrite', type=str2bool,
#                         default="False")
#     parser.add_argument('-r', '--crop', type=str,
#                         default="1", choices=config_dict["crop"])
#     parser.add_argument('-c', '--challenge', type=str,
#                         default="brats", choices=config_dict["challenge"],
#                         help="challenge name")
#     parser.add_argument('-y', '--year', type=int,
#                         default=2018, choices=config_dict["year"],
#                         help="year of challenge")
#     parser.add_argument('-is', '--image_shape', type=str,
#                         default="160-192-128", choices=config_dict["image_shape"],
#                         help="image shape to read")
#     parser.add_argument('-b', '--is_bias_correction', type=str,
#                         default="1", choices=config_dict["is_bias_correction"],
#                         help="perform bias field removal?")
#     parser.add_argument('-n', '--is_normalize', type=str,
#                         default="z", choices=config_dict["is_normalize"],
#                         help="what type of normalization")
#     parser.add_argument('-de', '--is_denoise', type=str,
#                         default="0", choices=config_dict["is_denoise"],
#                         help="what type of denoising")
#     parser.add_argument('-t', '--is_test', type=str,
#                         default="1", choices=["0", "1"])
#     parser.add_argument('-hi', '--is_hist_match', type=str,
#                         default="0")
#     parser.add_argument('-m', '--model', type=str,
#                         default="unet", choices=config_dict["model"])
#     parser.add_argument('-du', '--depth_unet', type=int,
#                         default=4, choices=config_dict["depth_unet"])
#     parser.add_argument('-nb', '--n_base_filters_unet', type=int,
#                         default=16, choices=config_dict["n_base_filters_unet"])
#     parser.add_argument('-ps', '--patch_shape', type=str,
#                         default="160-192-1",
#                         help="patch shape to train")
#     parser.add_argument('-crf', '--is_crf', type=str,
#                         default="0", choices=config_dict["is_crf"],
#                         help="crf method")
#     parser.add_argument('-ba', '--batch_size', type=int,
#                         default=64,
#                         help="train batch size")
#     parser.add_argument('-l', '--loss', type=str,
#                         default="weighted",
#                         help="loss function")
#     parser.add_argument('-alpha_tv', '--weight_tv_to_main_loss', type=float,
#                         default=0.1,
#                         help="weight of total variance comapared to main loss (dice coef etc.)")
#     args = parser.parse_args()
#     return args


# def train25d():
#     parser = argparse.ArgumentParser(description='Training')
#     parser.add_argument('-o', '--overwrite', type=str2bool,
#                         default="False")
#     parser.add_argument('-r', '--crop', type=str,
#                         default="1", choices=config_dict["crop"])
#     parser.add_argument('-c', '--challenge', type=str,
#                         default="brats", choices=config_dict["challenge"],
#                         help="challenge name")
#     parser.add_argument('-y', '--year', type=int,
#                         default=2018, choices=config_dict["year"],
#                         help="year of challenge")
#     parser.add_argument('-is', '--image_shape', type=str,
#                         default="160-192-128", choices=config_dict["image_shape"],
#                         help="image shape to read")
#     parser.add_argument('-b', '--is_bias_correction', type=str,
#                         default="1", choices=config_dict["is_bias_correction"],
#                         help="perform bias field removal?")
#     parser.add_argument('-n', '--is_normalize', type=str,
#                         default="z", choices=config_dict["is_normalize"],
#                         help="what type of normalization")
#     parser.add_argument('-de', '--is_denoise', type=str,
#                         default="0", choices=config_dict["is_denoise"],
#                         help="what type of denoising")
#     parser.add_argument('-t', '--is_test', type=str,
#                         default="1", choices=["0", "1"])
#     parser.add_argument('-hi', '--is_hist_match', type=str,
#                         default="0")
#     parser.add_argument('-m', '--model', type=str,
#                         default="unet")
#     parser.add_argument('-du', '--depth_unet', type=int,
#                         default=4, choices=config_dict["depth_unet"])
#     parser.add_argument('-nb', '--n_base_filters_unet', type=int,
#                         default=16, choices=config_dict["n_base_filters_unet"])
#     parser.add_argument('-ps', '--patch_shape', type=str,
#                         default="160-192-7",
#                         help="patch shape to train")
#     parser.add_argument('-crf', '--is_crf', type=str,
#                         default="0", choices=config_dict["is_crf"],
#                         help="crf method")
#     parser.add_argument('-ba', '--batch_size', type=int,
#                         default=64,
#                         help="train batch size")
#     parser.add_argument('-l', '--loss', type=str,
#                         default="weighted",
#                         help="loss function")
#     parser.add_argument('-alpha_tv', '--weight_tv_to_main_loss', type=float,
#                         default=0.1,
#                         help="weight of total variance comapared to main loss (dice coef etc.)")
#     args = parser.parse_args()
#     return args


# def prepare_data():
#     parser = argparse.ArgumentParser(description='Data preparation')
#     parser.add_argument('-o', '--overwrite', type=str2bool,
#                         default="True")
#     parser.add_argument('-r', '--crop', type=str,
#                         default="1", choices=config_dict["crop"])
#     parser.add_argument('-c', '--challenge', type=str,
#                         default="brats", choices=config_dict["challenge"],
#                         help="challenge name")
#     parser.add_argument('-y', '--year', type=str,
#                         default=2018, choices=config_dict["year"],
#                         help="year of challenge")
#     parser.add_argument('-is', '--image_shape', type=str,
#                         default="160-192-128", choices=config_dict["image_shape"],
#                         help="image shape to read")
#     parser.add_argument('-b', '--is_bias_correction', type=str,
#                         default="1", choices=config_dict["is_bias_correction"],
#                         help="perform bias field removal?")
#     parser.add_argument('-n', '--is_normalize', type=str,
#                         default="z", choices=config_dict["is_normalize"],
#                         help="what type of normalization")
#     parser.add_argument('-de', '--is_denoise', type=str,
#                         default="0", choices=config_dict["is_denoise"],
#                         help="what type of normalization")
#     parser.add_argument('-hi', '--is_hist_match', type=str,
#                         default="0")
#     parser.add_argument('-t', '--is_test', type=str,
#                         default="1", choices=["0", "1"])
#     args = parser.parse_args()
#     return args


# def train_ibsr():
#     parser = argparse.ArgumentParser(description='Training')
#     parser.add_argument('-o', '--overwrite', type=str2bool,
#                         default="False")
#     parser.add_argument('-r', '--crop', type=str,
#                         default="0", choices=config_dict["crop"])
#     parser.add_argument('-c', '--challenge', type=str,
#                         default="ibsr", choices=config_dict["challenge"],
#                         help="challenge name")
#     parser.add_argument('-y', '--year', type=str,
#                         default=2015, choices=config_dict["year"],
#                         help="year of challenge")
#     parser.add_argument('-is', '--image_shape', type=str,
#                         default="256-128-256", choices=config_dict["image_shape"],
#                         help="image shape to read")
#     parser.add_argument('-b', '--is_bias_correction', type=str,
#                         default="0", choices=config_dict["is_bias_correction"],
#                         help="perform bias field removal?")
#     parser.add_argument('-n', '--is_normalize', type=str,
#                         default="z", choices=config_dict["is_normalize"],
#                         help="what type of normalization")
#     parser.add_argument('-de', '--is_denoise', type=str,
#                         default="0", choices=config_dict["is_denoise"],
#                         help="what type of denoising")
#     parser.add_argument('-t', '--is_test', type=str,
#                         default="1", choices=["0", "1"])
#     parser.add_argument('-hi', '--is_hist_match', type=str,
#                         default="0")
#     parser.add_argument('-m', '--model', type=str,
#                         default="isensee", choices=config_dict["model"])
#     parser.add_argument('-du', '--depth_unet', type=int,
#                         default=4, choices=config_dict["depth_unet"])
#     parser.add_argument('-nb', '--n_base_filters_unet', type=int,
#                         default=16,
#                         # choices=config_dict["n_base_filters_unet"]
#                         )
#     parser.add_argument('-ps', '--patch_shape', type=str,
#                         default="128-128-128", choices=config_dict["patch_shape"],
#                         help="patch shape to train")
#     parser.add_argument('-crf', '--is_crf', type=str,
#                         default="0", choices=config_dict["is_crf"],
#                         help="crf method")
#     parser.add_argument('-ba', '--batch_size', type=int,
#                         default=1,
#                         help="train batch size")
#     parser.add_argument('-l', '--loss', type=str,
#                         default="weighted",
#                         help="loss function")
#     parser.add_argument('-dim', '--model_dim', type=int,
#                         default=3,
#                         help="2d, 2.5d or 3d?")
#     parser.add_argument('-alpha_tv', '--weight_tv_to_main_loss', type=float,
#                         default=0.1,
#                         help="weight of total variance comapared to main loss (dice coef etc.)")
#     args = parser.parse_args()
#     return args


# def prepare_data_ibsr():
#     parser = argparse.ArgumentParser(description='Data preparation')
#     parser.add_argument('-o', '--overwrite', type=str2bool,
#                         default="True")
#     parser.add_argument('-r', '--crop', type=str,
#                         default="0", choices=config_dict["crop"])
#     parser.add_argument('-c', '--challenge', type=str,
#                         default="ibsr", choices=config_dict["challenge"],
#                         help="challenge name")
#     parser.add_argument('-y', '--year', type=str,
#                         default=2015, choices=config_dict["year"],
#                         help="year of challenge")
#     parser.add_argument('-is', '--image_shape', type=str,
#                         default="256-128-256", choices=config_dict["image_shape"],
#                         help="image shape to read")
#     parser.add_argument('-b', '--is_bias_correction', type=str,
#                         default="0", choices=config_dict["is_bias_correction"],
#                         help="perform bias field removal?")
#     parser.add_argument('-n', '--is_normalize', type=str,
#                         default="z", choices=config_dict["is_normalize"],
#                         help="what type of normalization")
#     parser.add_argument('-de', '--is_denoise', type=str,
#                         default="0", choices=config_dict["is_denoise"],
#                         help="what type of normalization")
#     parser.add_argument('-hi', '--is_hist_match', type=str,
#                         default="0")
#     parser.add_argument('-t', '--is_test', type=str,
#                         default="1", choices=["0", "1"])
#     args = parser.parse_args()
#     return args
