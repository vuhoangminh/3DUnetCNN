import nibabel as nib
from unet3d.normalize_new import normalize_volume
from unet3d.utils.utils import save_nib

temp_volume_path = "/home/minhvu/Desktop/temp/volume.nii.gz"
temp_template_path = "/home/minhvu/Desktop/temp/template.nii.gz"
temp_volume_norm01_path = "/home/minhvu/Desktop/temp/norm01.nii.gz"
temp_volume_normz_path = "/home/minhvu/Desktop/temp/normz.nii.gz"
temp_volume_norm01_hist_path = "/home/minhvu/Desktop/temp/norm01_hist.nii.gz"
temp_volume_normz_hist_path = "/home/minhvu/Desktop/temp/normz_hist.nii.gz"

volume = nib.load(temp_volume_path)
volume = volume.get_data()
template = nib.load(temp_template_path)

affine = template.affine
template = template.get_data()
# save_nib(volume, temp_volume_path, affine)


volume_norm01 = normalize_volume(volume, template,
                                 is_normalize="01",
                                 is_hist_match="0")
save_nib(volume_norm01, temp_volume_norm01_path, affine)


volume_normz = normalize_volume(volume, template,
                                is_normalize="z",
                                is_hist_match="0")
save_nib(volume_normz, temp_volume_normz_path, affine)


volume_norm01_hist = normalize_volume(volume, template,
                                      is_normalize="01",
                                      is_hist_match="1")
save_nib(volume_norm01_hist, temp_volume_norm01_hist_path, affine)


volume_normz_hist = normalize_volume(volume, template,
                                     is_normalize="z",
                                     is_hist_match="1")
save_nib(volume_normz_hist, temp_volume_normz_hist_path, affine)


# save_nib(template, temp_template_path, affine)
# save_nib(volume_normalized, temp_volume_norm_path, affine)
# save_nib(volume_normalized-volume, temp_diff_path, affine)
