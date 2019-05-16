import numpy as np
from functools import partial
from keras import backend as K
import keras.layers as KL
from keras.engine import Input, Model
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Deconvolution2D, Dense, GlobalAveragePooling2D,
                          MaxPooling2D, Permute, PReLU, Reshape, UpSampling2D,
                          multiply)
from keras.optimizers import Adam
from unet3d.metrics import minh_dice_coef_metric
from keras.utils import multi_gpu_model
from unet3d.utils.model_utils import compile_model
from unet3d.model.unet_vae import GroupNormalization
from keras import regularizers


try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def baseline(inputs, pool_size=(2, 2), n_labels=1,
             deconvolution=False,
             depth=4, n_base_filters=32,
             batch_normalization=True,
             activation_name="sigmoid",
             is_unet_original=True,
             weight_decay=1e-5,
             name=None
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
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block2d(input_layer=current_layer,
                                            n_filters=n_base_filters *
                                            (2**layer_depth),
                                            batch_normalization=batch_normalization,
                                            weight_decay=weight_decay)
        layer2 = create_convolution_block2d(input_layer=layer1,
                                            n_filters=n_base_filters *
                                            (2**layer_depth)*2,
                                            batch_normalization=batch_normalization,
                                            weight_decay=weight_decay)
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
                                                   weight_decay=weight_decay)
        current_layer = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                   input_layer=current_layer,
                                                   batch_normalization=batch_normalization,
                                                   weight_decay=weight_decay)

    final_convolution = Conv2D(n_labels, (1, 1))(current_layer)
    output = Activation(activation_name, name=name)(final_convolution)

    return output


def casnet_v4(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False,
              batch_normalization=True, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-5
              ):

    # instead of concat, we multiply the output of previous mask
    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline(inp_whole, depth=depth, n_base_filters=n_base_filters,
                         weight_decay=weight_decay, name="out_whole")

    for i in range(inputs.shape[1]):
        if i < 1:
            mask_whole = out_whole
        else:
            mask_whole = concatenate([mask_whole, out_whole], axis=1)
    inp_core = multiply([mask_whole, inputs])
    out_core = baseline(inp_core, depth=depth, n_base_filters=n_base_filters,
                        weight_decay=weight_decay, name="out_core")

    for i in range(inputs.shape[1]):
        if i < 1:
            mask_core = out_core
        else:
            mask_core = concatenate([mask_core, out_core], axis=1)
    inp_core = multiply([mask_core, inputs])
    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline(inp_enh, depth=depth, n_base_filters=n_base_filters,
                       weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


def casnet_v3(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
              batch_normalization=False, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-5
              ):

    # each class (whole, core, enhance) is fed into different decoder branch
    inputs = Input(input_shape)

    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block2d(input_layer=current_layer,
                                            n_filters=n_base_filters *
                                            (2**layer_depth),
                                            batch_normalization=batch_normalization,
                                            weight_decay=weight_decay)
        layer2 = create_convolution_block2d(input_layer=layer1,
                                            n_filters=n_base_filters *
                                            (2**layer_depth)*2,
                                            batch_normalization=batch_normalization,
                                            weight_decay=weight_decay)
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    current_layer_whole = current_layer
    for layer_depth in range(depth-2, -1, -1):
        up_convolution_whole = get_up_convolution2d(pool_size=pool_size, deconvolution=deconvolution,
                                                    n_filters=current_layer_whole._keras_shape[1])(current_layer_whole)
        concat_whole = concatenate(
            [up_convolution_whole, levels[layer_depth][1]], axis=1)
        if not is_unet_original:
            concat_whole = squeeze_excite_block2d(concat_whole)
        current_layer_whole = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                         input_layer=concat_whole,
                                                         batch_normalization=batch_normalization,
                                                         weight_decay=weight_decay)
        current_layer_whole = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                         input_layer=current_layer_whole,
                                                         batch_normalization=batch_normalization,
                                                         weight_decay=weight_decay)

    final_convolution_whole = Conv2D(n_labels, (1, 1))(current_layer_whole)
    out_whole = Activation(activation_name, name="out_whole")(
        final_convolution_whole)

    # add levels with up-convolution or up-sampling
    current_layer_core = current_layer
    for layer_depth in range(depth-2, -1, -1):
        up_convolution_core = get_up_convolution2d(pool_size=pool_size, deconvolution=deconvolution,
                                                   n_filters=current_layer_core._keras_shape[1])(current_layer_core)

        if layer_depth == 0:
            concat_core = concatenate(
                [out_whole, up_convolution_core, levels[layer_depth][1]], axis=1)
        else:
            concat_core = concatenate(
                [up_convolution_core, levels[layer_depth][1]], axis=1)
        if not is_unet_original:
            concat_core = squeeze_excite_block2d(concat_core)
        current_layer_core = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                        input_layer=concat_core,
                                                        batch_normalization=batch_normalization,
                                                        weight_decay=weight_decay)
        current_layer_core = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                        input_layer=current_layer_core,
                                                        batch_normalization=batch_normalization,
                                                        weight_decay=weight_decay)

    final_convolution_core = Conv2D(n_labels, (1, 1))(current_layer_core)
    out_core = Activation(activation_name, name="out_core")(
        final_convolution_core)

    # add levels with up-convolution or up-sampling
    current_layer_enh = current_layer
    for layer_depth in range(depth-2, -1, -1):
        up_convolution_enh = get_up_convolution2d(pool_size=pool_size, deconvolution=deconvolution,
                                                  n_filters=current_layer_enh._keras_shape[1])(current_layer_enh)

        if layer_depth == 0:
            concat_enh = concatenate(
                [out_whole, out_core, up_convolution_core, levels[layer_depth][1]], axis=1)
        else:
            concat_enh = concatenate(
                [up_convolution_enh, levels[layer_depth][1]], axis=1)
        if not is_unet_original:
            concat_enh = squeeze_excite_block2d(concat_enh)
        current_layer_enh = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                       input_layer=concat_enh,
                                                       batch_normalization=batch_normalization,
                                                       weight_decay=weight_decay)
        current_layer_enh = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                       input_layer=current_layer_enh,
                                                       batch_normalization=batch_normalization,
                                                       weight_decay=weight_decay)

    final_convolution_enh = Conv2D(n_labels, (1, 1))(current_layer_enh)
    out_enh = Activation(activation_name, name="out_enh")(
        final_convolution_enh)

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


def casnet_v2(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False,
              batch_normalization=True, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-5
              ):

    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline(inp_whole, depth=depth, n_base_filters=n_base_filters,
                         weight_decay=weight_decay, name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline(inp_core, depth=depth, n_base_filters=n_base_filters,
                        weight_decay=weight_decay, name="out_core")

    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline(inp_enh, depth=depth, n_base_filters=n_base_filters,
                       weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


def casnet_v1(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
              batch_normalization=False, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-5
              ):

    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline(inp_whole, depth=4, n_base_filters=16,
                         weight_decay=weight_decay, name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline(inp_core, depth=3, n_base_filters=16,
                        weight_decay=weight_decay, name="out_core")

    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline(inp_enh, depth=2, n_base_filters=16,
                       weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         #  metrics=metrics,
                         initial_learning_rate=initial_learning_rate)


def sepnet_v1(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
              batch_normalization=False, activation_name="sigmoid",
              loss_function="sepweighted",
              is_unet_original=True,
              weight_decay=1e-5
              ):

    # each class (whole, core, enhance) is fed into different decoder branch
    inputs = Input(input_shape)

    current_layer = inputs

    current_layer = squeeze_excite_block2d(current_layer)
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block2d(input_layer=current_layer,
                                            n_filters=n_base_filters *
                                            (2**layer_depth),
                                            batch_normalization=batch_normalization,
                                            weight_decay=weight_decay)
        layer2 = create_convolution_block2d(input_layer=layer1,
                                            n_filters=n_base_filters *
                                            (2**layer_depth)*2,
                                            batch_normalization=batch_normalization,
                                            weight_decay=weight_decay)
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    current_layer_1 = current_layer
    for layer_depth in range(depth-2, -1, -1):
        up_convolution_1 = get_up_convolution2d(pool_size=pool_size, deconvolution=deconvolution,
                                                n_filters=current_layer_1._keras_shape[1])(current_layer_1)
        concat_1 = concatenate(
            [up_convolution_1, levels[layer_depth][1]], axis=1)
        concat_1 = squeeze_excite_block2d(concat_1)
        current_layer_1 = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                     input_layer=concat_1,
                                                     batch_normalization=batch_normalization,
                                                     weight_decay=weight_decay)
        current_layer_1 = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                     input_layer=current_layer_1,
                                                     batch_normalization=batch_normalization,
                                                     weight_decay=weight_decay)

    final_convolution_1 = Conv2D(n_labels, (1, 1))(current_layer_1)
    out_1 = Activation(activation_name, name="out_1")(
        final_convolution_1)

    # add levels with up-convolution or up-sampling
    current_layer_2 = current_layer
    for layer_depth in range(depth-2, -1, -1):
        up_convolution_2 = get_up_convolution2d(pool_size=pool_size, deconvolution=deconvolution,
                                                n_filters=current_layer_2._keras_shape[1])(current_layer_2)

        if layer_depth == 0:
            concat_2 = concatenate(
                [up_convolution_2, levels[layer_depth][1]], axis=1)
        else:
            concat_2 = concatenate(
                [up_convolution_2, levels[layer_depth][1]], axis=1)
        concat_2 = squeeze_excite_block2d(concat_2)
        current_layer_2 = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                     input_layer=concat_2,
                                                     batch_normalization=batch_normalization,
                                                     weight_decay=weight_decay)
        current_layer_2 = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                     input_layer=current_layer_2,
                                                     batch_normalization=batch_normalization,
                                                     weight_decay=weight_decay)

    final_convolution_2 = Conv2D(n_labels, (1, 1))(current_layer_2)
    out_2 = Activation(activation_name, name="out_2")(
        final_convolution_2)

    # add levels with up-convolution or up-sampling
    current_layer_4 = current_layer
    for layer_depth in range(depth-2, -1, -1):
        up_convolution_4 = get_up_convolution2d(pool_size=pool_size, deconvolution=deconvolution,
                                                n_filters=current_layer_4._keras_shape[1])(current_layer_4)

        if layer_depth == 0:
            concat_4 = concatenate(
                [up_convolution_2, levels[layer_depth][1]], axis=1)
        else:
            concat_4 = concatenate(
                [up_convolution_4, levels[layer_depth][1]], axis=1)
        concat_4 = squeeze_excite_block2d(concat_4)
        current_layer_4 = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                     input_layer=concat_4,
                                                     batch_normalization=batch_normalization,
                                                     weight_decay=weight_decay)
        current_layer_4 = create_convolution_block2d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                     input_layer=current_layer_4,
                                                     batch_normalization=batch_normalization,
                                                     weight_decay=weight_decay)

    final_convolution_4 = Conv2D(n_labels, (1, 1))(current_layer_4)
    out_4 = Activation(activation_name, name="out_4")(
        final_convolution_4)

    model = Model(inputs=inputs, outputs=[out_1, out_2, out_4])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


def create_convolution_block2d(input_layer, n_filters, batch_normalization=False, kernel=(3, 3), activation=None,
                               padding='same', strides=(1, 1), instance_normalization=False,
                               is_unet_original=True, weight_decay=0):
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
    layer = Conv2D(n_filters, kernel,
                   padding=padding,
                   strides=strides,
                   kernel_regularizer=regularizers.l2(l=weight_decay))(input_layer)
    if batch_normalization:
        # layer = BatchNormalization(axis=1)(layer)
        layer = GroupNormalization(groups=16, axis=1)(layer)
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
        layer = squeeze_excite_block2d(layer)
    return layer


def compute_level_output_shape2d(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 2d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(
        np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution2d(n_filters, pool_size, kernel_size=(2, 2), strides=(2, 2),
                         deconvolution=False):
    if deconvolution:
        return Deconvolution2D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling2D(size=pool_size)


def squeeze_excite_block2d(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]

    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(max(filters//ratio, 1), activation='relu',
               kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid',
               kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def main():
    model = sepnet_v1(input_shape=(4, 160, 192))
    model.summary()


if __name__ == "__main__":
    main()
