from unet3d.normalize import perform_clahe
import nibabel as nib
from unet3d.normalize import normalize_volume
from unet3d.utils.utils import save_nib


from unet3d.utils.path_utils import get_workspace_path

is_desktop = False

save_dir = get_workspace_path(is_desktop)

temp_volume_path = save_dir + "volume.nii.gz"
temp_template_path = save_dir + "template.nii.gz"
temp_volume_norm01_path = save_dir + "norm01.nii.gz"
temp_volume_normz_path = save_dir + "normz.nii.gz"
temp_volume_norm01_hist_path = save_dir + "norm01_hist.nii.gz"
temp_volume_normz_hist_path = save_dir + "normz_hist.nii.gz"
temp_volume_clahe_path = save_dir + "volume_clahe.nii.gz"

volume = nib.load(temp_volume_path)
volume = volume.get_data()
template = nib.load(temp_template_path)

affine = template.affine
template = template.get_data()
# save_nib(volume, temp_volume_path, affine)


# volume_norm01 = normalize_volume(volume, template,
#                                  is_normalize="01",
#                                  is_hist_match="0")
# save_nib(volume_norm01, temp_volume_norm01_path, affine)


# volume_normz = normalize_volume(volume, template,
#                                 is_normalize="z",
#                                 is_hist_match="0")
# save_nib(volume_normz, temp_volume_normz_path, affine)


# volume_norm01_hist = normalize_volume(volume, template,
#                                       is_normalize="01",
#                                       is_hist_match="1")
# save_nib(volume_norm01_hist, temp_volume_norm01_hist_path, affine)


# volume_normz_hist = normalize_volume(volume, template,
#                                      is_normalize="z",
#                                      is_hist_match="1")
# save_nib(volume_normz_hist, temp_volume_normz_hist_path, affine)


import numpy as np
source = volume

# reshape to 1d
H, W, D = source.shape
source_2d = source.reshape((H, W*D))


volume_clahe = perform_clahe(source_2d, clip_limit=0.002)
volume_clahe = volume_clahe.reshape(source.shape)
save_nib(volume_clahe, temp_volume_clahe_path, affine)


# save_nib(template, temp_template_path, affine)
# save_nib(volume_normalized, temp_volume_norm_path, affine)
# save_nib(volume_normalized-volume, temp_diff_path, affine)
