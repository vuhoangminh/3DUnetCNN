import os
import numpy as np

from keras.utils import plot_model
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import multi_gpu_model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.models import Model
import tensorflow as tf

from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
from unet3d.metrics import minh_dice_coef_loss, dice_coefficient_loss, minh_dice_coef_metric
from unet3d.metrics import weighted_dice_coefficient_loss, soft_dice_loss, soft_dice_numpy, tversky_loss
from unet3d.metrics import tv_minh_loss

from unet3d.model.unet import get_up_convolution

""" 
Generates the FCN architechture
Adapted from the Caffe implementation at:
https://github.com/mrkolarik/3D-brain-segmentation
"""

K.set_image_data_format('channels_first')


def res_unet_3d(input_shape, n_labels=1,
                initial_learning_rate=0.00001, deconvolution=False,
                pool_size=(2, 2, 2),
                n_base_filters=32, include_label_wise_dice_coefficients=False,
                batch_normalization=False, activation_name="sigmoid",
                metrics=minh_dice_coef_metric,
                loss_function="minh"):
    if K.image_data_format() == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = 4
    inputs = Input(input_shape)
    conv1 = Conv3D(n_base_filters, (3, 3, 3),
                   activation='relu', padding='same')(inputs)
    conv1 = Conv3D(n_base_filters, (3, 3, 3),
                   activation='relu', padding='same')(conv1)
    conc1 = concatenate([inputs, conv1], axis=concat_axis)
    pool1 = MaxPooling3D(pool_size=pool_size)(conc1)

    conv2 = Conv3D(n_base_filters*2, (3, 3, 3),
                   activation='relu', padding='same')(pool1)
    conv2 = Conv3D(n_base_filters*2, (3, 3, 3),
                   activation='relu', padding='same')(conv2)
    conc2 = concatenate([pool1, conv2], axis=concat_axis)
    pool2 = MaxPooling3D(pool_size=pool_size)(conc2)

    conv3 = Conv3D(n_base_filters*4, (3, 3, 3),
                   activation='relu', padding='same')(pool2)
    conv3 = Conv3D(n_base_filters*4, (3, 3, 3),
                   activation='relu', padding='same')(conv3)
    conc3 = concatenate([pool2, conv3], axis=concat_axis)
    pool3 = MaxPooling3D(pool_size=pool_size)(conc3)

    conv4 = Conv3D(n_base_filters*8, (3, 3, 3),
                   activation='relu', padding='same')(pool3)
    conv4 = Conv3D(n_base_filters*8, (3, 3, 3),
                   activation='relu', padding='same')(conv4)
    conc4 = concatenate([pool3, conv4], axis=concat_axis)
    pool4 = MaxPooling3D(pool_size=pool_size)(conc4)

    conv5 = Conv3D(n_base_filters*16, (3, 3, 3),
                   activation='relu', padding='same')(pool4)
    conv5 = Conv3D(n_base_filters*16, (3, 3, 3),
                   activation='relu', padding='same')(conv5)
    conc5 = concatenate([pool4, conv5], axis=concat_axis)

    up6 = concatenate([Conv3DTranspose(n_base_filters*8, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conc5), conv4], axis=concat_axis)
    conv6 = Conv3D(n_base_filters*8, (3, 3, 3),
                   activation='relu', padding='same')(up6)
    conv6 = Conv3D(n_base_filters*8, (3, 3, 3),
                   activation='relu', padding='same')(conv6)
    conc6 = concatenate([up6, conv6], axis=concat_axis)

    up7 = concatenate([Conv3DTranspose(n_base_filters*4, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conc6), conv3], axis=concat_axis)
    conv7 = Conv3D(n_base_filters*4, (3, 3, 3),
                   activation='relu', padding='same')(up7)
    conv7 = Conv3D(n_base_filters*4, (3, 3, 3),
                   activation='relu', padding='same')(conv7)
    conc7 = concatenate([up7, conv7], axis=concat_axis)

    up8 = concatenate([Conv3DTranspose(n_base_filters*2, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conc7), conv2], axis=concat_axis)
    conv8 = Conv3D(n_base_filters*2, (3, 3, 3),
                   activation='relu', padding='same')(up8)
    conv8 = Conv3D(n_base_filters*2, (3, 3, 3),
                   activation='relu', padding='same')(conv8)
    conc8 = concatenate([up8, conv8], axis=concat_axis)

    up9 = concatenate([Conv3DTranspose(n_base_filters, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conc8), conv1], axis=concat_axis)
    conv9 = Conv3D(n_base_filters, (3, 3, 3),
                   activation='relu', padding='same')(up9)
    conv9 = Conv3D(n_base_filters, (3, 3, 3),
                   activation='relu', padding='same')(conv9)
    conc9 = concatenate([up9, conv9], axis=concat_axis)

    conv10 = Conv3D(n_labels, (1, 1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(
            index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
    try:
        model = multi_gpu_model(model, gpus=2)
        print('!! train on multi gpus')
    except:
        print('!! train on single gpu')
        pass

    if loss_function == "tversky":
        loss = tversky_loss
    elif loss_function == "minh":
        loss = minh_dice_coef_loss
    elif loss_function == "tv_minh":
        loss = tv_minh_loss
    else:
        loss = weighted_dice_coefficient_loss

    model.compile(optimizer=Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999),
                  loss=loss, metrics=metrics)
    return model


def dense_unet_3d(input_shape, n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False,
                  pool_size=(2, 2, 2),
                  n_base_filters=32, include_label_wise_dice_coefficients=False,
                  batch_normalization=False, activation_name="sigmoid",
                  metrics=minh_dice_coef_metric,
                  loss_function="minh"):
    if K.image_data_format() == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = 4
    inputs = Input(input_shape)
    conv11 = Conv3D(n_base_filters, (3, 3, 3), activation='relu', padding='same')(inputs)
    conc11 = concatenate([inputs, conv11], axis=concat_axis)
    conv12 = Conv3D(n_base_filters, (3, 3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=concat_axis)
    pool1 = MaxPooling3D(pool_size=pool_size)(conc12)

    conv21 = Conv3D(n_base_filters*2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=concat_axis)
    conv22 = Conv3D(n_base_filters*2, (3, 3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=concat_axis)
    pool2 = MaxPooling3D(pool_size=pool_size)(conc22)

    conv31 = Conv3D(n_base_filters*4, (3, 3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=concat_axis)
    conv32 = Conv3D(n_base_filters*4, (3, 3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=concat_axis)
    pool3 = MaxPooling3D(pool_size=pool_size)(conc32)

    conv41 = Conv3D(n_base_filters*8, (3, 3, 3), activation='relu', padding='same')(pool3)
    conc41 = concatenate([pool3, conv41], axis=concat_axis)
    conv42 = Conv3D(n_base_filters*8, (3, 3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([pool3, conv42], axis=concat_axis)
    pool4 = MaxPooling3D(pool_size=pool_size)(conc42)

    conv51 = Conv3D(n_base_filters*16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conc51 = concatenate([pool4, conv51], axis=concat_axis)
    conv52 = Conv3D(n_base_filters*16, (3, 3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool4, conv52], axis=concat_axis)

    up6 = concatenate([Conv3DTranspose(n_base_filters*8, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conc52), conc42], axis=concat_axis)
    conv61 = Conv3D(n_base_filters*8, (3, 3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=concat_axis)
    conv62 = Conv3D(n_base_filters*8, (3, 3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=concat_axis)

    up7 = concatenate([Conv3DTranspose(n_base_filters*4, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conc62), conv32], axis=concat_axis)
    conv71 = Conv3D(n_base_filters*4, (3, 3, 3), activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=concat_axis)
    conv72 = Conv3D(n_base_filters*4, (3, 3, 3), activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=concat_axis)

    up8 = concatenate([Conv3DTranspose(n_base_filters*2, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conc72), conv22], axis=concat_axis)
    conv81 = Conv3D(n_base_filters*2, (3, 3, 3), activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=concat_axis)
    conv82 = Conv3D(n_base_filters*2, (3, 3, 3), activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=concat_axis)

    up9 = concatenate([Conv3DTranspose(n_base_filters, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conc82), conv12], axis=concat_axis)
    conv91 = Conv3D(n_base_filters, (3, 3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=concat_axis)
    conv92 = Conv3D(n_base_filters, (3, 3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=concat_axis)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(
            index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics
    try:
        model = multi_gpu_model(model, gpus=2)
        print('!! train on multi gpus')
    except:
        print('!! train on single gpu')
        pass

    if loss_function == "tversky":
        loss = tversky_loss
    elif loss_function == "minh":
        loss = minh_dice_coef_loss
    elif loss_function == "tv_minh":
        loss = tv_minh_loss
    else:
        loss = weighted_dice_coefficient_loss

    model.compile(optimizer=Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999),
                  loss=loss, metrics=metrics)
    return model
