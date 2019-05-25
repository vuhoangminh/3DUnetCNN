from functools import partial


from keras.layers.core import Dense, Lambda
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, MaxPooling3D
from keras.layers import SpatialDropout3D, Conv3D, BatchNormalization, Dropout
from keras.engine import Model
from keras.optimizers import Adam

from .unet import concatenate
from ..metrics import weighted_dice_coefficient_loss, tversky_loss
from ..metrics import minh_dice_coef_loss, minh_dice_coef_metric

from keras.utils import multi_gpu_model
from keras import regularizers
from keras.layers.merge import concatenate, add
from unet3d.utils.model_utils import compile_model
from unet3d.model.unet_vae import GroupNormalization
import keras.backend as K

# import tensorflow as tf
# import external.gradient_checkpointing.memory_saving_gradients as memory_saving_gradients
# # from tensorflow.python.keras._impl.keras import backend as K
# import tensorflow.keras.backend as K
# # K.__dict__["gradients"] = memory_saving_gradients.gradients_memory
# # K.__dict__["gradients"] = memory_saving_gradients.gradients_speed


def pernet(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
           n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
           loss_function="weighted", activation_name="sigmoid", metrics=minh_dice_coef_metric,
           labels=[1, 2, 4]):
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

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(
                current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number])
        concatenation_layer = concatenate(
            [level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(
                0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         labels=labels,
                         initial_learning_rate=initial_learning_rate)


def mnet(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
         n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
         weight_decay=1e-5, loss_function="weighted", activation_name="sigmoid",
         metrics=minh_dice_coef_metric, labels=[1, 2, 4]):
    """
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
    inp = Input(input_shape)
    # The Initial Block

    # The Initial Block
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        padding='same',
        data_format='channels_first',
        name='Input_x1')(inp)

    ## Dropout (0.2)
    x = Dropout(0.2)(x)

    # Green Block x1 (output filters = 32)
    # x1 = __bottleneck_block(x, 32, 4)
    x1 = green_block(x, 32, weight_decay=weight_decay, name='x1')
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=2,
        kernel_regularizer=regularizers.l2(weight_decay),
        padding='same',
        data_format='channels_first',
        name='Enc_DownSample_32')(x1)

    # Green Block x2 (output filters = 64)
    x = green_block(x, 64, weight_decay=weight_decay, name='Enc_64_1')
    x2 = green_block(x, 64, weight_decay=weight_decay, name='x2')
    x = Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        strides=2,
        kernel_regularizer=regularizers.l2(weight_decay),
        padding='same',
        data_format='channels_first',
        name='Enc_DownSample_64')(x2)

    # Green Blocks x2 (output filters = 128)
    x = green_block(x, 128, weight_decay=weight_decay, name='Enc_128_1')
    x3 = green_block(x, 128, weight_decay=weight_decay, name='x3')
    x = Conv3D(
        filters=128,
        kernel_size=(3, 3, 3),
        strides=2,
        kernel_regularizer=regularizers.l2(weight_decay),
        padding='same',
        data_format='channels_first',
        name='Enc_DownSample_128')(x3)

    # Green Blocks x4 (output filters = 256)
    x = green_block(x, 256, weight_decay=weight_decay, name='Enc_256_1')
    x = green_block(x, 256, weight_decay=weight_decay, name='Enc_256_2')
    x = green_block(x, 256, weight_decay=weight_decay, name='Enc_256_3')
    x4 = green_block(x, 256, weight_decay=weight_decay, name='x4')

    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------

    # GT (Groud Truth) Part
    # -------------------------------------------------------------------------

    # Green Block x1 (output filters=128)
    x = Conv3D(
        filters=128,
        kernel_size=(1, 1, 1),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        data_format='channels_first',
        name='Dec_GT_ReduceDepth_128')(x4)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_GT_UpSample_128')(x)
    x = Add(name='Input_Dec_GT_128')([x, x3])
    x = green_block(x, 128, weight_decay=weight_decay, name='Dec_GT_128')

    # Green Block x1 (output filters=64)
    x = Conv3D(
        filters=64,
        kernel_size=(1, 1, 1),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        data_format='channels_first',
        name='Dec_GT_ReduceDepth_64')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_GT_UpSample_64')(x)
    x = Add(name='Input_Dec_GT_64')([x, x2])
    x = green_block(x, 64, weight_decay=weight_decay, name='Dec_GT_64')

    # Green Block x1 (output filters=32)
    x = Conv3D(
        filters=32,
        kernel_size=(1, 1, 1),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        data_format='channels_first',
        name='Dec_GT_ReduceDepth_32')(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Dec_GT_UpSample_32')(x)
    x = Add(name='Input_Dec_GT_32')([x, x1])
    x = green_block(x, 32, weight_decay=weight_decay, name='Dec_GT_32')

    # Blue Block x1 (output filters=32)
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        padding='same',
        data_format='channels_first',
        name='Input_Dec_GT_Output')(x)

    # Output Block
    out_GT = Conv3D(
        filters=n_labels,  # No. of tumor classes is 3
        kernel_size=(1, 1, 1),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        data_format='channels_first',
        activation='sigmoid',
        name='Dec_GT_Output')(x)

    # Build and Compile the model
    out = out_GT
    model = Model(inp, out)  # Create the model

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         labels=labels,
                         initial_learning_rate=initial_learning_rate)


def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
                   if K.image_data_format() == 'channels_last' else
                   lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = Activation('relu')(x)

    return x


def __bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    ''' Adds a bottleneck block
    Args:
        input: input tensor
        filters: number of output filters
        cardinality: cardinality factor described number of
            grouped convolutions
        strides: performs strided convolution for downsampling if > 1
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    init = input

    grouped_channels = int(filters / cardinality)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 2 * filters:
            init = Conv3D(filters * 2, (1, 1, 1), padding='same', strides=(strides, strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:
        if init._keras_shape[-1] != 2 * filters:
            init = Conv3D(filters * 2, (1, 1, 1), padding='same', strides=(strides, strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = Conv3D(filters, (1, 1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = __grouped_convolution_block(
        x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv3D(filters * 2, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    x = add([init, x])
    x = Activation('relu')(x)

    return x


def green_block_new(inp, filters, data_format='channels_first',
                    cardinality=8, strides=1, weight_decay=5e-4,
                    name=None):
    """
    green_block(inp, filters, name=None)
    ------------------------------------
    Implementation of the special residual block used in the paper. The block
    consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
    units, with a residual connection from the input `inp` to the output. Used
    internally in the model. Can be used independently as well.

    Parameters
    ----------
    `inp`: An keras.layers.layer instance, required
        The keras layer just preceding the green block.
    `filters`: integer, required
        No. of filters to use in the 3D convolutional block. The output
        layer of this green block will have this many no. of channels.
    `data_format`: string, optional
        The format of the input data. Must be either 'chanels_first' or
        'channels_last'. Defaults to `channels_first`, as used in the paper.
    `name`: string, optional
        The name to be given to this green block. Defaults to None, in which
        case, keras uses generated names for the involved layers. If a string
        is provided, the names of individual layers are generated by attaching
        a relevant prefix from [GroupNorm_, Res_, Conv3D_, Relu_, ], followed
        by _1 or _2.

    Returns
    -------
    `out`: A keras.layers.Layer instance
        The output of the green block. Has no. of channels equal to `filters`.
        The size of the rest of the dimensions remains same as in `inp`.
    """
    channel_axis = 1 if data_format == 'channels_first' else -1
    grouped_channels = int(filters / cardinality)

    inp_res = Conv3D(
        filters=filters,
        kernel_size=(1, 1, 1),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        data_format=data_format,
        name=f'Res_{name}' if name else None)(inp)

    # axis=1 for channels_first data format
    # No. of groups = 8, as given in the paper
    x = BatchNormalization(
        axis=channel_axis,
        name=f'BatchNorm_1_{name}' if name else None)(inp)
    x = Activation('relu', name=f'Relu_1_{name}' if name else None)(x)
    x = Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        padding='same',
        data_format=data_format,
        name=f'Conv3D_1_{name}' if name else None)(x)

    x = __grouped_convolution_block(
        x, grouped_channels, cardinality, strides, weight_decay)

    x = BatchNormalization(
        axis=channel_axis,
        name=f'BatchNorm_2_{name}' if name else None)(x)
    x = Activation('relu', name=f'Relu_2_{name}' if name else None)(x)
    x = Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name=f'Conv3D_2_{name}' if name else None)(x)

    out = Add(name=f'Out_{name}' if name else None)([x, inp_res])
    return out


def green_block(inp, filters, data_format='channels_first', weight_decay=5e-4, name=None):
    """
    green_block(inp, filters, name=None)
    ------------------------------------
    Implementation of the special residual block used in the paper. The block
    consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
    units, with a residual connection from the input `inp` to the output. Used
    internally in the model. Can be used independently as well.

    Parameters
    ----------
    `inp`: An keras.layers.layer instance, required
        The keras layer just preceding the green block.
    `filters`: integer, required
        No. of filters to use in the 3D convolutional block. The output
        layer of this green block will have this many no. of channels.
    `data_format`: string, optional
        The format of the input data. Must be either 'chanels_first' or
        'channels_last'. Defaults to `channels_first`, as used in the paper.
    `name`: string, optional
        The name to be given to this green block. Defaults to None, in which
        case, keras uses generated names for the involved layers. If a string
        is provided, the names of individual layers are generated by attaching
        a relevant prefix from [GroupNorm_, Res_, Conv3D_, Relu_, ], followed
        by _1 or _2.

    Returns
    -------
    `out`: A keras.layers.Layer instance
        The output of the green block. Has no. of channels equal to `filters`.
        The size of the rest of the dimensions remains same as in `inp`.
    """
    inp_res = Conv3D(
        filters=filters,
        kernel_size=(1, 1, 1),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        data_format=data_format,
        name=f'Res_{name}' if name else None)(inp)

    # axis=1 for channels_first data format
    # No. of groups = 8, as given in the paper
    x = GroupNormalization(
        groups=8,
        axis=1 if data_format == 'channels_first' else 0,
        name=f'GroupNorm_1_{name}' if name else None)(inp)
    x = Activation('relu', name=f'Relu_1_{name}' if name else None)(x)
    x = Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        padding='same',
        data_format=data_format,
        name=f'Conv3D_1_{name}' if name else None)(x)

    x = GroupNormalization(
        groups=8,
        axis=1 if data_format == 'channels_first' else 0,
        name=f'GroupNorm_2_{name}' if name else None)(x)
    x = Activation('relu', name=f'Relu_2_{name}' if name else None)(x)
    x = Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        kernel_regularizer=regularizers.l2(weight_decay),
        padding='same',
        data_format=data_format,
        name=f'Conv3D_2_{name}' if name else None)(x)

    out = Add(name=f'Out_{name}' if name else None)([x, inp_res])
    return out


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=LeakyReLU,
                             padding='same', strides=(1, 1, 1), normalization="Batch"):
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
    layer = Conv3D(n_filters, kernel,
                   padding=padding,
                   strides=strides,
                   # kernel_regularizer=regularizers.regularizers.l2(l=1e-4))(input_layer)#doesn't work
                   )(input_layer)
    if normalization == " Batch":
        layer = BatchNormalization(axis=1)(layer)
    elif normalization == " Instance":
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    else:
        layer = GroupNormalization(groups=8, axis=1)(layer)
    if activation is None:
        layer = Activation('relu')(layer)
    else:
        layer = activation()(layer)
    return layer


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
