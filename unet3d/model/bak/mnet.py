from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation
from keras.layers import SpatialDropout3D, Conv3D, Deconvolution3D, MaxPooling3D
from keras.engine import Model
from keras.optimizers import Adam

from .unet import create_convolution_block, concatenate, get_up_convolution, squeeze_excite_block
from ..metrics import weighted_dice_coefficient_loss, tversky_loss, minh_dice_coef_loss, minh_dice_coef_metric

from keras.utils import multi_gpu_model
from unet3d.utils.model_utils import compile_model

create_convolution_block = partial(
    create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def simple_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1,
                    initial_learning_rate=0.00001,
                    activation_name="sigmoid",
                    depth=10,
                    n_base_filters=32,
                    metrics=minh_dice_coef_metric,
                    loss_function="weighted",
                    labels=[1, 2, 4],
                    is_unet_original=True):
    inputs = Input(input_shape)
    current_layer = inputs
    for layer_depth in range(depth):
        current_layer = create_convolution_block(input_layer=current_layer,
                                                 n_filters=n_base_filters,
                                                 batch_normalization=False,
                                                 is_unet_original=is_unet_original)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         labels=labels,
                         initial_learning_rate=initial_learning_rate)


def eye_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1,
                 initial_learning_rate=0.00001,
                 activation_name="sigmoid",
                 depth=5,
                 n_base_filters=8,
                 growth_rate=2,
                 metrics=minh_dice_coef_metric,
                 loss_function="weighted",
                 labels=[1, 2, 4],
                 is_unet_original=True):
    inputs = Input(input_shape)
    current_layer = inputs
    for layer_depth in reversed(range(depth)):
        kernel_size = 3 + layer_depth*growth_rate
        n_filters = n_base_filters*2**(depth-layer_depth-1)
        current_layer = create_convolution_block(input_layer=current_layer,
                                                 n_filters=n_filters,
                                                 batch_normalization=False,
                                                 is_unet_original=is_unet_original,
                                                 kernel=(kernel_size, kernel_size, kernel_size))

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         labels=labels,
                         initial_learning_rate=initial_learning_rate)


def mnet_model_3d(input_shape=(4, 128, 128, 128), n_base_filters=32,
                  pool_size=(2, 2, 2),
                  n_segmentation_levels=3, n_labels=4,
                  initial_learning_rate=5e-4,
                  labels=[1, 2, 4],
                  loss_function="weighted", activation_name="sigmoid",
                  metrics=minh_dice_coef_metric):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 challenge:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    down0 = inputs
    down1 = create_convolution_block(down0, n_filters=4, strides=pool_size)
    down2 = create_convolution_block(down1, n_filters=4, strides=pool_size)
    down3 = create_convolution_block(down2, n_filters=4, strides=pool_size)

    conv0 = create_convolution_block(down0, n_filters=n_base_filters)
    conv1 = create_convolution_block(down1, n_filters=n_base_filters)
    conv2 = create_convolution_block(down2, n_filters=n_base_filters)
    conv3 = create_convolution_block(down3, n_filters=n_base_filters)

    out3 = conv3
    out3 = UpSampling3D(size=pool_size)(out3)

    concat2 = concatenate([out3, conv2], axis=1)
    out2 = create_convolution_block(concat2, n_filters=n_base_filters)
    out2 = create_up_sampling_module(
        out2, n_filters=n_base_filters, size=pool_size)

    concat1 = concatenate([out2, conv1], axis=1)
    out1 = create_convolution_block(concat1, n_filters=n_base_filters)
    out1 = create_up_sampling_module(
        out1, n_filters=n_base_filters, size=pool_size)

    concat0 = concatenate([out1, conv0], axis=1)
    out0 = create_convolution_block(concat0, n_filters=n_base_filters)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(out0)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         labels=labels,
                         initial_learning_rate=initial_learning_rate)


def mnet_model2_3d(input_shape=(4, 128, 128, 128), n_base_filters=32,
                   pool_size=(2, 2, 2),
                   n_segmentation_levels=3, n_labels=4,
                   initial_learning_rate=5e-4,
                   loss_function="weighted", activation_name="sigmoid",
                   metrics=minh_dice_coef_metric):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 challenge:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    down0 = inputs
    down1 = create_convolution_block(down0, n_filters=4, strides=pool_size)
    down2 = create_convolution_block(down1, n_filters=4, strides=pool_size)
    down3 = create_convolution_block(down2, n_filters=4, strides=pool_size)
    # down1 = MaxPooling3D(pool_size=pool_size)(down0)
    # down2 = MaxPooling3D(pool_size=pool_size)(down1)
    # down3 = MaxPooling3D(pool_size=pool_size)(down2)

    conv0 = create_convolution_block(down0, n_filters=n_base_filters)
    conv1 = create_convolution_block(down1, n_filters=n_base_filters*2)
    conv2 = create_convolution_block(down2, n_filters=n_base_filters*4)
    conv3 = create_convolution_block(down3, n_filters=n_base_filters*8)

    out3 = conv3
    out3 = UpSampling3D(size=pool_size)(out3)

    concat2 = concatenate([out3, conv2], axis=1)
    out2 = create_convolution_block(concat2, n_filters=n_base_filters*4)
    out2 = create_up_sampling_module2(
        out2, n_filters=n_base_filters*4, size=pool_size)

    concat1 = concatenate([out2, conv1], axis=1)
    out1 = create_convolution_block(concat1, n_filters=n_base_filters*2)
    out1 = create_up_sampling_module2(
        out1, n_filters=n_base_filters*2, size=pool_size)

    concat0 = concatenate([out1, conv0], axis=1)
    out0 = create_convolution_block(concat0, n_filters=n_base_filters)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(out0)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         initial_learning_rate=initial_learning_rate)


def multiscale_unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                             depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
                             batch_normalization=False, activation_name="sigmoid",
                             metrics=minh_dice_coef_metric,
                             loss_function="weighted",
                             is_unet_original=True,
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
    levels = list()

    down_levels = list()
    # add levels with max pooling
    for layer_depth in range(depth):
        layer = create_convolution_block(current_layer, n_filters=n_base_filters *
                                         (2**layer_depth), strides=pool_size)
        current_layer = layer
        down_levels.append([layer])

    current_layer = inputs
    for layer_depth in range(depth):
        if layer_depth > 0:
            down_layer = down_levels[layer_depth-1][0]
            current_layer = concatenate([down_layer, current_layer], axis=1)

        layer1 = create_convolution_block(input_layer=current_layer,
                                          n_filters=n_base_filters *
                                          (2**layer_depth),
                                          batch_normalization=batch_normalization,
                                          is_unet_original=is_unet_original)

        layer2 = create_convolution_block(input_layer=layer1,
                                          n_filters=n_base_filters *
                                          (2**layer_depth)*2,
                                          batch_normalization=batch_normalization,
                                          is_unet_original=is_unet_original)
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
        if not is_unet_original:
            concat = squeeze_excite_block(concat)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat,
                                                 batch_normalization=batch_normalization,
                                                 is_unet_original=is_unet_original)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization,
                                                 is_unet_original=is_unet_original)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         labels=labels,
                         initial_learning_rate=initial_learning_rate)


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(
        convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_up_sampling_module2(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = Deconvolution3D(filters=n_filters, kernel_size=size,
                                strides=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution
