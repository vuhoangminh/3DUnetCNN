from unet2d.utils.patches import compute_patch_indices
from unet2d.generator import create_patch_index_list

image_shape=(160,192,128)
patch_size=(160,192,1)
# patch_size=(128,128,128)
indices = compute_patch_indices(image_shape, patch_size, overlap=0,
                          start=None, 
                          is_extract_patch_agressive=False,
                          is_predict=False)

indices = create_patch_index_list([0,1,2], image_shape, patch_size, 0)

print(indices)