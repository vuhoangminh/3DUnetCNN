import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def display_array_as_image(array):
    plt.imshow(array, cmap="gray")
    plt.show()
    # img = Image.fromarray(array)
    # img.show()
