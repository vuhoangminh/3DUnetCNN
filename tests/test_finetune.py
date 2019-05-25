from unet3d.utils.model_utils import generate_model
from unet3d.training import load_old_model

laptop_save_dir = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/database/model/done/minh_loss/"
desktop_save_dir = "/home/minhvu/github/3DUnetCNN_BRATS/brats/database/model/base/"
save_dir = laptop_save_dir
save_dir = desktop_save_dir
model_path = save_dir + "brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-1_ps-128-128-128_unet_crf-0_d-4_nb-16_loss-weighted_model.h5"

model_path = save_dir + "brats_2018_is-160-192-128_crop-1_bias-1_denoise-0_norm-01_hist-0_ps-128-128-128_isensee_crf-0_loss-weighted_model.h5"

# model = load_old_model(model_path)
# model.summary()

a = 2

from unet3d.metrics import weighted_dice_coefficient_loss, tversky_loss, minh_dice_coef_loss, minh_dice_coef_metric

model = load_old_model(model_path)
# model = generate_model(model_path, loss_function="weighted",
#                        metrics=minh_dice_coef_metric,
# labels=config["labels"],
#                        initial_learning_rate=0.001)
model.summary()
# weights = model.
