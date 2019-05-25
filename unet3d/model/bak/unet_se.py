import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.layers import GlobalAveragePooling3D, Reshape, Dense, multiply, Permute
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
from unet3d.metrics import minh_dice_coef_loss, dice_coefficient_loss, minh_dice_coef_metric
from unet3d.metrics import weighted_dice_coefficient_loss, soft_dice_loss, soft_dice_numpy, tversky_loss
from unet3d.metrics import tv_minh_loss
from unet3d.utils.model_utils import compile_model

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def se_unet_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
               depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
               batch_normalization=False, activation_name="sigmoid",
               metrics=minh_dice_coef_metric,
               loss_function="weighted",
               labels=[1, 2, 4]
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
    # current_layer = squeeze_excite_block(current_layer)
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_se_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                             batch_normalization=batch_normalization)
        layer2 = create_convolution_se_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                             batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        concat = squeeze_excite_block(concat)
        current_layer = create_convolution_se_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                    input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_se_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                    input_layer=current_layer,
                                                    batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    # if not isinstance(metrics, list):
    #     metrics = [metrics]

    # if include_label_wise_dice_coefficients and n_labels > 1:
    #     label_wise_dice_metrics = [get_label_dice_coefficient_function(
    #         index) for index in range(n_labels)]
    #     if metrics:
    #         metrics = metrics + label_wise_dice_metrics
    #     else:
    #         metrics = label_wise_dice_metrics

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         labels=labels,
                         initial_learning_rate=initial_learning_rate)


def create_convolution_se_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                                padding='same', strides=(1, 1, 1), instance_normalization=True):
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
    layer = Conv3D(n_filters, kernel, padding=padding,
                   strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        layer = Activation('relu')(layer)
    else:
        layer = activation()(layer)

    layer = squeeze_excite_block(layer)
    return layer


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=True,
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
    layer = Conv3D(n_filters, kernel, padding=padding,
                   strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        layer = Activation('relu')(layer)
    else:
        layer = activation()(layer)
    if not is_unet_original:
        layer = squeeze_excite_block(layer)
    return layer


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(
        np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)


def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]

    # se_shape = (filters, 1, 1, 1) if K.image_data_format() == "channels_first" else (1, 1, 1, filters)
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(max(filters//ratio, 1), activation='relu',
               kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid',
               kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    x = multiply([init, se])
    return x
