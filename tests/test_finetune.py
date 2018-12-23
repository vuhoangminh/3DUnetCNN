from unet3d.training import load_old_model

laptop_save_dir = "C:/Users/minhm/Documents/GitHub/3DUnetCNN_BRATS/brats/database/model/done/minh_loss/"
desktop_save_dir = "/home/minhvu/Desktop/temp/"
save_dir = laptop_save_dir
model_path = save_dir + "brats_2018_is-160-192-128_crop-1_bias-1_denoise-bm4d_norm-01_hist-1_ps-128-128-128_unet_crf-0_d-4_nb-16_model.h5"

model = load_old_model(model_path)
model.summary()

a = 2

# weights = model.