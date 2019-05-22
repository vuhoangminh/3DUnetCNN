import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import (Activation, BatchNormalization, Conv2D, Conv3D,
                          Deconvolution2D, Dense, GlobalAveragePooling2D,
                          MaxPooling2D, Permute, PReLU, Reshape, ZeroPadding3D,
                          multiply)

from unet2d.model.blocks import (compute_level_output_shape2d,
                                 create_convolution_block2d,
                                 get_up_convolution2d, squeeze_excite_block2d)
from unet3d.metrics import minh_dice_coef_metric
from unet3d.model.unet import create_convolution_block
from unet3d.utils.model_utils import compile_model

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_25d(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                   depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
                   batch_normalization=False, activation_name="sigmoid",
                   metrics=minh_dice_coef_metric,
                   loss_function="weighted",
                   is_unet_original=True
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
    levels = list()

    current_layer = create_transition_3d_to_2d(input_layer=inputs,
                                               n_filters=n_base_filters,
                                               is_unet_original=is_unet_original,
                                               batch_normalization=batch_normalization,
                                               instance_normalization=False)

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block2d(input_layer=current_layer,
                                            n_filters=n_base_filters *
                                            (2**layer_depth),
                                            batch_normalization=batch_normalization,
                                            is_unet_original=is_unet_original)
        layer2 = create_convolution_block2d(input_layer=layer1,
                                            n_filters=n_base_filters *
                                            (2**layer_depth)*2,
                                            batch_normalization=batch_normalization,
                                            is_unet_original=is_unet_original)
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution2d(pool_size=pool_size, deconvolution=deconvolution,
                                              n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        if not is_unet_original:
            concat = squeeze_excite_block2d(concat)
        current_layer = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                   input_layer=concat,
                                                   batch_normalization=batch_normalization,
                                                   is_unet_original=is_unet_original)
        current_layer = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                   input_layer=current_layer,
                                                   batch_normalization=batch_normalization,
                                                   is_unet_original=is_unet_original)

    final_convolution = Conv2D(n_labels, (1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         initial_learning_rate=initial_learning_rate)


def create_transition_3d_to_2d(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3),
                               activation=None, padding='valid', strides=(1, 1, 1),
                               instance_normalization=False,
                               is_unet_original=True):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    shape = input_layer._keras_shape

    height = shape[-1]
    kernel_size = kernel[0]

    if height % (kernel_size-1) != 1:
        raise ValueError(
            "mod(depth_inputs, kernel_size) should be 1. Please check!")

    depth = height//(kernel_size-1)

    current_layer = input_layer
    for _ in range(depth):
        current_layer = ZeroPadding3D(
            padding=(1, 1, 0), data_format="channels_first")(current_layer)
        current_layer = create_convolution_block(input_layer=current_layer, n_filters=n_filters,
                                                 batch_normalization=batch_normalization,
                                                 is_unet_original=is_unet_original,
                                                 instance_normalization=instance_normalization,
                                                 padding=padding)

    shape = current_layer._keras_shape
    to_shape = shape[1:len(shape)-1]

    current_layer = Reshape(to_shape)(current_layer)
    return current_layer