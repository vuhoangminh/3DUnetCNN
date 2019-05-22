from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling2D, Activation, SpatialDropout2D, Conv2D
from keras.engine import Model
from keras.optimizers import Adam

from unet2d.model.unet_2d import create_convolution_block2d, concatenate
from unet2d.model.isensee2d import create_localization_module
from unet2d.model.isensee2d import create_context_module
from unet2d.model.isensee2d import create_up_sampling_module


from unet25d.model.unet25d import create_transition_3d_to_2d
from unet3d.metrics import minh_dice_coef_metric

from unet3d.utils.model_utils import compile_model

create_convolution_block2d = partial(
    create_convolution_block2d, activation=LeakyReLU, instance_normalization=True)


def isensee25d_model(input_shape=(4, 128, 128, 7), n_base_filters=16, depth=5, dropout_rate=0.3,
                     n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                     loss_function="weighted", activation_name="sigmoid", metrics=minh_dice_coef_metric,
                     is_unet_original=True):
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

    first_layer = create_transition_3d_to_2d(input_layer=inputs,
                                             n_filters=n_base_filters,
                                             is_unet_original=is_unet_original,
                                             instance_normalization=False)

    current_layer = first_layer

    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is first_layer:
            in_conv = create_convolution_block2d(
                current_layer, n_level_filters,
                is_unet_original=is_unet_original)
        else:
            in_conv = create_convolution_block2d(
                current_layer, n_level_filters, strides=(2, 2),
                is_unet_original=is_unet_original)

        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate,
            is_unet_original=is_unet_original)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number], is_unet_original=is_unet_original)
        concatenation_layer = concatenate(
            [level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number], is_unet_original=is_unet_original)
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(
                0, Conv2D(n_labels, (1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling2D(size=(2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         initial_learning_rate=initial_learning_rate)
