import os
import numpy as np

from keras.utils import plot_model
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import multi_gpu_model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D, Activation
from keras.models import Model
import tensorflow as tf

from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
from unet3d.metrics import minh_dice_coef_loss, dice_coefficient_loss, minh_dice_coef_metric
from unet3d.metrics import weighted_dice_coefficient_loss, soft_dice_loss, soft_dice_numpy, tversky_loss
from unet3d.metrics import tv_minh_loss
from unet3d.utils.model_utils import compile_model

from unet3d.model.unet import get_up_convolution
from unet3d.model.unet import create_convolution_block
from unet3d.model.unet import compute_level_output_shape

""" 
Generates the FCN architechture
Adapted from the Caffe implementation at:
https://github.com/mrkolarik/3D-brain-segmentation
"""

K.set_image_data_format('channels_first')


def res_unet_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
                batch_normalization=False, activation_name="sigmoid",
                metrics=minh_dice_coef_metric,
                loss_function="weighted"
                ):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        concat_layer = concatenate([current_layer, layer2], axis=1)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(concat_layer)
            levels.append([layer1, layer2, concat_layer, current_layer])
        else:
            current_layer = concat_layer
            levels.append([layer1, layer2, concat_layer])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
        current_layer = concatenate([concat, current_layer], axis=1)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         initial_learning_rate=initial_learning_rate)


def dense_unet_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
                  batch_normalization=False, activation_name="sigmoid",
                  metrics=minh_dice_coef_metric,
                  loss_function="weighted"
                  ):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        concat1_layer = concatenate([current_layer, layer1], axis=1)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        concat2_layer = concatenate([current_layer, layer2], axis=1)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(concat2_layer)
            levels.append([layer1, concat1_layer, layer2,
                           concat2_layer, current_layer])
        else:
            current_layer = concat2_layer
            levels.append([layer1, layer2, concat2_layer])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][2]], axis=1)
        layer1 = create_convolution_block(n_filters=levels[layer_depth][2]._keras_shape[1],
                                          input_layer=concat, batch_normalization=batch_normalization)
        current_layer = concatenate([concat, layer1], axis=1)
        layer2 = create_convolution_block(n_filters=levels[layer_depth][2]._keras_shape[1],
                                          input_layer=current_layer,
                                          batch_normalization=batch_normalization)
        current_layer = concatenate([concat, layer2], axis=1)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         initial_learning_rate=initial_learning_rate)