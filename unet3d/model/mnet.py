from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam

from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss, tversky_loss, minh_dice_coef_loss, minh_dice_coef_metric

from keras.utils import multi_gpu_model
from unet3d.utils.model_utils import compile_model

create_convolution_block = partial(
    create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def mnet_model_3d(input_shape=(4, 128, 128, 128), n_base_filters=32,
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
    down1 = create_convolution_block(down0, n_filters=4, strides=(2, 2, 2))
    down2 = create_convolution_block(down1, n_filters=4, strides=(2, 2, 2))
    down3 = create_convolution_block(down2, n_filters=4, strides=(2, 2, 2))

    conv0 = create_convolution_block(down0, n_filters=n_base_filters)
    conv1 = create_convolution_block(down1, n_filters=n_base_filters)
    conv2 = create_convolution_block(down2, n_filters=n_base_filters)
    conv3 = create_convolution_block(down3, n_filters=n_base_filters)

    out3 = conv3
    out3 = UpSampling3D(size=pool_size)(out3)

    concat2 = concatenate([out3, conv2], axis=1)
    out2 = create_convolution_block(concat2, n_filters=n_base_filters)
    out2 = create_up_sampling_module(out2, n_filters=n_base_filters, size=(2, 2, 2))

    concat1 = concatenate([out2, conv1], axis=1)
    out1 = create_convolution_block(concat1, n_filters=n_base_filters)
    out1 = create_up_sampling_module(out1, n_filters=n_base_filters, size=(2, 2, 2))

    concat0 = concatenate([out1, conv0], axis=1)
    out0 = create_convolution_block(concat0, n_filters=n_base_filters)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(out0)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
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


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(
        input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(
        rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(
        input_layer=dropout, n_filters=n_level_filters)
    return convolution2
