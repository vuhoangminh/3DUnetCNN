import numpy as np
import nibabel as nib
from unet3d.metrics import tv_ndim_loss

from keras import backend as K
K.set_image_data_format("channels_first")

# laptop_save_dir = "C:/Users/minhm/Desktop/temp/"
# desktop_save_dir = "/home/minhvu/Desktop/temp/"
# save_dir = laptop_save_dir

# temp_volume_path = save_dir + "volume.nii.gz"
# temp_template_path = save_dir + "template.nii.gz"
# gaussian_path = save_dir + "gaussian.nii.gz"
# truth_path = save_dir + "truth.nii.gz"
# prediction_path = save_dir + "prediction.nii.gz"

# volume = nib.load(temp_volume_path)
# affine = volume.affine
# volume = volume.get_data()
# template = nib.load(temp_template_path)
# template = template.get_data()
# truth = nib.load(truth_path)
# truth = truth.get_data()

# prediction = nib.load(prediction_path)
# prediction = prediction.get_data()


x = np.array((1, 2, 240, 240, 155))

x = K.variable(x)

y = K.update()


# # x[0,0,:,:,:] = volume
# # x[0,1,:,:,:] = template

# x = truth
# loss = K.eval(tv_ndim_loss(K.variable(x)))
# print(loss)

# x = prediction
# loss = K.eval(tv_ndim_loss(K.variable(x)))
# print(loss)

