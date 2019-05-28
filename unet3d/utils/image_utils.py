import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def display_array_as_image(array):
    plt.imshow(array, cmap="gray")
    plt.show()


def move_axis_data(list_data, source, destination):
    new_list_data = list()
    for i in range(len(list_data)):
        data = list_data[i]
        data = np.moveaxis(data, source=source, destination=destination)
        new_list_data.append(data)
    return new_list_data
