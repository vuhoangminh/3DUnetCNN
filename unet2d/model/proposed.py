from keras.engine import Input, Model
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Deconvolution2D, Dense, GlobalAveragePooling2D,
                          MaxPooling2D, Permute, PReLU, Reshape, UpSampling2D,
                          multiply, Add)
from unet3d.utils.model_utils import compile_model
from unet2d.model.blocks import get_up_convolution2d
from unet2d.model.blocks import squeeze_excite_block2d
from unet2d.model.blocks import create_convolution_block2d, conv_block_resnet2d

from keras.layers.merge import concatenate


# U-Net 2D with conv_block
def baseline_unet(inputs, pool_size=(2, 2), n_labels=1,
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


# Isensee 2D with conv_block
def baseline_isensee(inputs, n_base_filters=16, depth=5,
                     dropout_rate=0.3, n_segmentation_levels=3, n_labels=1,
                     weight_decay=1e-5,
                     activation_name="sigmoid",
                     name=None):

    from unet2d.model.isensee2d import create_context_module
    from unet2d.model.isensee2d import create_up_sampling_module
    from unet2d.model.isensee2d import create_localization_module

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block2d(
                current_layer, n_level_filters,
                weight_decay=weight_decay)
        else:
            in_conv = create_convolution_block2d(
                current_layer, n_level_filters, strides=(2, 2),
                weight_decay=weight_decay)

        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate,
            weight_decay=weight_decay)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number],
            weight_decay=weight_decay)
        concatenation_layer = concatenate(
            [level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number],
            weight_decay=weight_decay)
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

    output = Activation(activation_name, name=name)(output_layer)
    return output


# U-Net 2D with resnet_block
def baseline_resnet(inputs, pool_size=(2, 2), n_labels=1,
                    deconvolution=False,
                    depth=4, n_base_filters=16,
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
        n_filters = n_base_filters * (2**layer_depth)
        layer1 = conv_block_resnet2d(input_layer=current_layer,
                                     kernel_size=3,
                                     n_filters=[n_filters,
                                                n_filters, n_filters],
                                     stage=layer_depth,
                                     block=name+"_en_a",
                                     weight_decay=weight_decay
                                     )
        layer2 = conv_block_resnet2d(input_layer=layer1,
                                     kernel_size=3,
                                     n_filters=[n_filters,
                                                n_filters, n_filters],
                                     stage=layer_depth,
                                     block=name+"_en_b",
                                     weight_decay=weight_decay
                                     )
        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        n_filters = levels[layer_depth][1]._keras_shape[1]
        up_convolution = get_up_convolution2d(pool_size=pool_size, deconvolution=deconvolution,
                                              n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)

        current_layer = conv_block_resnet2d(input_layer=concat,
                                            kernel_size=3,
                                            n_filters=[n_filters,
                                                       n_filters, n_filters],
                                            stage=layer_depth,
                                            block=name+"_de_a",
                                            weight_decay=weight_decay
                                            )

        current_layer = conv_block_resnet2d(input_layer=current_layer,
                                            kernel_size=3,
                                            n_filters=[n_filters,
                                                       n_filters, n_filters],
                                            stage=layer_depth,
                                            block=name+"_de_b",
                                            weight_decay=weight_decay
                                            )

    final_convolution = Conv2D(n_labels, (1, 1))(current_layer)
    output = Activation(activation_name, name=name)(final_convolution)

    return output


# casnet_v9 (similar to v2): replace conv_block by baseline_isensee
def casnet_v9(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False,
              batch_normalization=True, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-4
              ):

    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline_isensee(inp_whole, depth=depth, n_base_filters=n_base_filters,
                                 weight_decay=weight_decay, name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline_isensee(inp_core, depth=depth, n_base_filters=n_base_filters,
                                weight_decay=weight_decay, name="out_core")

    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline_isensee(inp_enh, depth=depth, n_base_filters=n_base_filters,
                               weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


# casnet_v8 (similar to v2): replace regularizer 1e-5 to 1e-4
def casnet_v8(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False,
              batch_normalization=True, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-4
              ):

    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline_unet(inp_whole, depth=depth, n_base_filters=n_base_filters,
                              weight_decay=weight_decay, name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline_unet(inp_core, depth=depth, n_base_filters=n_base_filters,
                             weight_decay=weight_decay, name="out_core")

    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline_unet(inp_enh, depth=depth, n_base_filters=n_base_filters,
                            weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


# casnet_v7 (similar to v2): replace regularizer 1e-5 to 1e-3
def casnet_v7(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False,
              batch_normalization=True, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-3
              ):

    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline_unet(inp_whole, depth=depth, n_base_filters=n_base_filters,
                              weight_decay=weight_decay, name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline_unet(inp_core, depth=depth, n_base_filters=n_base_filters,
                             weight_decay=weight_decay, name="out_core")

    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline_unet(inp_enh, depth=depth, n_base_filters=n_base_filters,
                            weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


# casnet_v6 (similar to v2): replace conv_block by resnet_block
def casnet_v6(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
              batch_normalization=False, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-5
              ):

    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline_resnet(inp_whole, depth=depth, n_base_filters=n_base_filters,
                                weight_decay=weight_decay, name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline_resnet(inp_core, depth=depth, n_base_filters=n_base_filters,
                               weight_decay=weight_decay, name="out_core")

    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline_resnet(inp_enh, depth=depth, n_base_filters=n_base_filters,
                              weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


# casnet_v5: 3 2D U-Net: whole_d5_n16, core_d5_n16, enh_d5_n16
def casnet_v5(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False,
              batch_normalization=True, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-5
              ):

    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline_unet(inp_whole, depth=5, n_base_filters=16,
                              weight_decay=weight_decay, name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline_unet(inp_core, depth=5, n_base_filters=16,
                             weight_decay=weight_decay, name="out_core")

    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline_unet(inp_enh, depth=5, n_base_filters=16,
                            weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


# casnet_v4 (similar to v2): inputs of core and enh = multiplication of inputs and "masks"
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
    out_whole = baseline_unet(inp_whole, depth=depth, n_base_filters=n_base_filters,
                              weight_decay=weight_decay, name="out_whole")

    for i in range(inputs.shape[1]):
        if i < 1:
            mask_whole = out_whole
        else:
            mask_whole = concatenate([mask_whole, out_whole], axis=1)
    inp_core = multiply([mask_whole, inputs])
    out_core = baseline_unet(inp_core, depth=depth, n_base_filters=n_base_filters,
                             weight_decay=weight_decay, name="out_core")

    for i in range(inputs.shape[1]):
        if i < 1:
            mask_core = out_core
        else:
            mask_core = concatenate([mask_core, out_core], axis=1)
    inp_core = multiply([mask_core, inputs])
    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline_unet(inp_enh, depth=depth, n_base_filters=n_base_filters,
                            weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


# casnet_v3: 1 encoder, 3 decoders
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


# casnet_v2: 3 2D U-Net: whole_d4_n16, core_d4_n16, enh_d4_n16
def casnet_v2(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False,
              batch_normalization=True, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-5
              ):

    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline_unet(inp_whole, depth=depth, n_base_filters=n_base_filters,
                              weight_decay=weight_decay, name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline_unet(inp_core, depth=depth, n_base_filters=n_base_filters,
                             weight_decay=weight_decay, name="out_core")

    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline_unet(inp_enh, depth=depth, n_base_filters=n_base_filters,
                            weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate)


# casnet_v1: 3 2D U-Net: whole_d4_n16, core_d3_n16, enh_d2_n16
def casnet_v1(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
              batch_normalization=False, activation_name="sigmoid",
              loss_function="casweighted",
              is_unet_original=True,
              weight_decay=1e-5
              ):

    inputs = Input(input_shape)
    inp_whole = inputs
    out_whole = baseline_unet(inp_whole, depth=4, n_base_filters=16,
                              weight_decay=weight_decay, name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline_unet(inp_core, depth=3, n_base_filters=16,
                             weight_decay=weight_decay, name="out_core")

    inp_enh = concatenate([out_core, inp_core], axis=1)
    out_enh = baseline_unet(inp_enh, depth=2, n_base_filters=16,
                            weight_decay=weight_decay, name="out_enh")

    model = Model(inputs=inputs, outputs=[out_whole, out_core, out_enh])

    return compile_model(model, loss_function=loss_function,
                         #  metrics=metrics,
                         initial_learning_rate=initial_learning_rate)


# sepnet_v1: 1 encoder, 3 decoders for 3 classes (1, 2 ,4) + w/ squeeze_excite_block2d after concat
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


# sepnet_v1: 1 encoder, 3 decoders for 3 classes (1, 2 ,4) + wo/ squeeze_excite_block2d after concat
def sepnet_v2(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
              depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
              batch_normalization=False, activation_name="sigmoid",
              loss_function="sepweighted",
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
    current_layer_1 = current_layer
    for layer_depth in range(depth-2, -1, -1):
        up_convolution_1 = get_up_convolution2d(pool_size=pool_size, deconvolution=deconvolution,
                                                n_filters=current_layer_1._keras_shape[1])(current_layer_1)
        concat_1 = concatenate(
            [up_convolution_1, levels[layer_depth][1]], axis=1)
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
        concat_2 = concatenate(
            [up_convolution_2, levels[layer_depth][1]], axis=1)
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
        concat_4 = concatenate(
            [up_convolution_4, levels[layer_depth][1]], axis=1)
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


def main():
    model = casnet_v9(input_shape=(4, 160, 192))
    model.summary()


if __name__ == "__main__":
    main()
