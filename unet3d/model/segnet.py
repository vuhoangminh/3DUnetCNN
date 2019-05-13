import os
import numpy as np
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.applications import imagenet_utils

# https://github.com/namakemono/keras-segnet/blob/master/segnet.py


def SegNet(input_shape=(360, 480, 3), classes=12):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution3D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Convolution3D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Convolution3D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Convolution3D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Decoder
    x = Convolution3D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Convolution3D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Convolution3D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Convolution3D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution3D(classes, 1, 1, border_mode="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model
