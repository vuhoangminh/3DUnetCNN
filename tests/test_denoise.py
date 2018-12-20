import nibabel as nib
from unet3d.utils.utils import save_nib
from scipy.ndimage.filters import gaussian_filter, median_filter

laptop_save_dir = "C:/Users/minhm/Desktop/temp/"
desktop_save_dir = "/home/minhvu/Desktop/temp/"
save_dir = desktop_save_dir

temp_volume_path = save_dir + "volume.nii.gz"
temp_template_path = save_dir + "template.nii.gz"
gaussian_path = save_dir + "gaussian.nii.gz"
median_path = save_dir + "median.nii.gz"

volume = nib.load(temp_volume_path)
affine = volume.affine
volume = volume.get_data()

gaussian = gaussian_filter(volume, sigma=0.5)
median = median_filter(volume, size=3)

save_nib(gaussian, gaussian_path, affine)
save_nib(median, median_path, affine)