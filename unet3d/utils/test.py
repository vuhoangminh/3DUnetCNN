# import os

# os.chdir("C://Users//minhm//Documents//GitHub//3DUnetCNN//unet3d")

from patches import compute_patch_indices

print("Hello world")

image_shape = (240, 240, 155)  # This determines what shape the images will be cropped/resampled to.
patch_size = (64, 64, 64)  # switch to None to train on the whole image
a = compute_patch_indices(image_shape, patch_size, 0, start=None)


print(a)



print("-"*60)
print("")
print("-"*60)