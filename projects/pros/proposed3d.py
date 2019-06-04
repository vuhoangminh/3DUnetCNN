from keras.engine import Input, Model
from keras.layers import (Activation, BatchNormalization, Conv3D,
                          Deconvolution3D, Dense, GlobalAveragePooling3D,
                          MaxPooling3D, Permute, PReLU, Reshape, UpSampling3D,
                          multiply, Add)

from unet3d.model.blocks import get_down_sampling
from unet3d.model.blocks import get_up_convolution3d
from unet3d.model.blocks import squeeze_excite_block3d
from unet3d.model.blocks import create_convolution_block3d, conv_block_resnet3d

from keras.layers.merge import concatenate

from keras.utils import multi_gpu_model
from unet3d.training import load_old_model
from tensorflow.python.client import device_lib
import os
from keras.models import model_from_json
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam

from unet3d.metrics import weighted_dice_coefficient_loss
from unet3d.metrics import tversky_loss
from unet3d.metrics import minh_dice_coef_loss
from unet3d.metrics import tv_minh_loss
from unet3d.metrics import tv_weighted_loss
from unet3d.metrics import minh_dice_coef_metric
from unet3d.metrics import dice_coefficient_loss
from keras.losses import categorical_crossentropy


from keras.layers.merge import concatenate


# U-Net 3D with conv_block
def baseline_unet(inputs, pool_size=(2, 2, 2), n_labels=1,
                  deconvolution=False,
                  depth=4, n_base_filters=32,
                  batch_normalization=True,
                  activation_name="sigmoid",
                  is_unet_original=True,
                  weight_decay=1e-5,
                  name=None,
                  down_levels=None
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
        if down_levels is not None and layer_depth > 0:
            current_layer = concatenate(
                [down_levels[layer_depth], current_layer], axis=1)
        layer1 = create_convolution_block3d(input_layer=current_layer,
                                            n_filters=n_base_filters *
                                            (2**layer_depth),
                                            batch_normalization=batch_normalization,
                                            weight_decay=weight_decay)
        layer2 = create_convolution_block3d(input_layer=layer1,
                                            n_filters=n_base_filters *
                                            (2**layer_depth)*2,
                                            batch_normalization=batch_normalization,
                                            weight_decay=weight_decay)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution3d(pool_size=pool_size, deconvolution=deconvolution,
                                              n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        if not is_unet_original:
            concat = squeeze_excite_block3d(concat)
        current_layer = create_convolution_block3d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                   input_layer=concat,
                                                   batch_normalization=batch_normalization,
                                                   weight_decay=weight_decay)
        current_layer = create_convolution_block3d(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                   input_layer=current_layer,
                                                   batch_normalization=batch_normalization,
                                                   weight_decay=weight_decay)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    output = Activation(activation_name, name=name)(final_convolution)

    return output


# Isensee 3D with conv_block
def baseline_isensee(inputs, n_base_filters=16, depth=5,
                     dropout_rate=0.3, n_segmentation_levels=3, n_labels=1,
                     weight_decay=1e-5,
                     activation_name="sigmoid",
                     name=None,
                     down_levels=None):

    from unet3d.model.isensee3d import create_context_module3d
    from unet3d.model.isensee3d import create_up_sampling_module3d
    from unet3d.model.isensee3d import create_localization_module3d

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for layer_depth in range(depth):
        n_level_filters = (2**layer_depth) * n_base_filters
        level_filters.append(n_level_filters)

        if down_levels is not None and layer_depth > 0:
            current_layer = concatenate(
                [down_levels[layer_depth], current_layer], axis=1)

        if current_layer is inputs:
            in_conv = create_convolution_block3d(
                current_layer, n_level_filters,
                weight_decay=weight_decay)
        else:
            in_conv = create_convolution_block3d(
                current_layer, n_level_filters, strides=(2, 2, 2),
                weight_decay=weight_decay)

        context_output_layer = create_context_module3d(
            in_conv, n_level_filters, dropout_rate=dropout_rate,
            weight_decay=weight_decay)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for layer_depth in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module3d(
            current_layer, level_filters[layer_depth],
            weight_decay=weight_decay)
        concatenation_layer = concatenate(
            [level_output_layers[layer_depth], up_sampling], axis=1)
        localization_output = create_localization_module3d(
            concatenation_layer, level_filters[layer_depth],
            weight_decay=weight_decay)
        current_layer = localization_output
        if layer_depth < n_segmentation_levels:
            segmentation_layers.insert(
                0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for layer_depth in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[layer_depth]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if layer_depth > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    output = Activation(activation_name, name=name)(output_layer)
    return output


# U-Net 3D with resnet_block
def baseline_resnet(inputs, pool_size=(2, 2, 2), n_labels=1,
                    deconvolution=False,
                    depth=4, n_base_filters=16,
                    batch_normalization=True,
                    activation_name="sigmoid",
                    is_unet_original=True,
                    weight_decay=1e-5,
                    name=None,
                    down_levels=None
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

        if down_levels is not None and layer_depth > 0:
            current_layer = concatenate(
                [down_levels[layer_depth], current_layer], axis=1)

        layer1 = conv_block_resnet3d(input_layer=current_layer,
                                     kernel_size=3,
                                     n_filters=[n_filters,
                                                n_filters, n_filters],
                                     stage=layer_depth,
                                     block=name+"_en_a",
                                     weight_decay=weight_decay
                                     )
        layer2 = conv_block_resnet3d(input_layer=layer1,
                                     kernel_size=3,
                                     n_filters=[n_filters,
                                                n_filters, n_filters],
                                     stage=layer_depth,
                                     block=name+"_en_b",
                                     weight_decay=weight_decay
                                     )
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        n_filters = levels[layer_depth][1]._keras_shape[1]
        up_convolution = get_up_convolution3d(pool_size=pool_size, deconvolution=deconvolution,
                                              n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)

        current_layer = conv_block_resnet3d(input_layer=concat,
                                            kernel_size=3,
                                            n_filters=[n_filters,
                                                       n_filters, n_filters],
                                            stage=layer_depth,
                                            block=name+"_de_a",
                                            weight_decay=weight_decay
                                            )

        current_layer = conv_block_resnet3d(input_layer=current_layer,
                                            kernel_size=3,
                                            n_filters=[n_filters,
                                                       n_filters, n_filters],
                                            stage=layer_depth,
                                            block=name+"_de_b",
                                            weight_decay=weight_decay
                                            )

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    output = Activation(activation_name, name=name)(final_convolution)

    return output


# casnet_v10 (similar to v2): concat down-sampled input at different levels of encoders
def casnet_v10(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
               depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False,
               batch_normalization=True, activation_name="sigmoid",
               loss_function="casweighted",
               weight_decay=1e-5, labels=[1, 2]
               ):

    inputs = Input(input_shape)
    inp_whole = inputs
    down_levels = get_down_sampling(inputs, depth=depth)
    out_whole = baseline_unet(inp_whole, depth=depth,
                              n_base_filters=n_base_filters,
                              weight_decay=weight_decay,
                              down_levels=down_levels,
                              name="out_whole")

    inp_core = concatenate([out_whole, inp_whole], axis=1)
    out_core = baseline_unet(inp_core, depth=depth,
                             n_base_filters=n_base_filters,
                             weight_decay=weight_decay,
                             down_levels=down_levels,
                             name="out_core")

    model = Model(inputs=inputs, outputs=[out_whole, out_core])

    return compile_model(model, loss_function=loss_function,
                         initial_learning_rate=initial_learning_rate,
                         labels=labels)


def load_model_multi_gpu(model_file):

    print(">> load old model")
    model = load_old_model(model_file)

    from unet3d.utils.path_utils import get_filename
    filename = get_filename(model_file)

    model_json_path = filename.replace(".h5", ".json")
    weights_path = filename
    if os.path.exists(model_json_path):
        print(">> remove old json")
        os.remove(model_json_path)
    if os.path.exists(weights_path):
        print(">> remove old weights")
        os.remove(weights_path)

    # ------------ save the template model rather than the gpu_mode ----------------
    # serialize model to JSON
    print(">> save architecture to disk")
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    print(">> save weights to disk")
    model.save_weights(weights_path)

    # -------------- load the saved model --------------
    from keras.models import model_from_json

    # load json and create model
    print(">> load architecture from disk")
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print(">> load model from disk")
    loaded_model.load_weights(weights_path)

    for i_layer in range(len(loaded_model.layers)):
        layer_name = loaded_model.layers[i_layer].name
        if type(loaded_model.layers[i_layer]) is Model:
            model = loaded_model.layers[i_layer]

    print(">> remove temp weights")
    os.remove(weights_path)
    print(">> remove temp json")
    os.remove(model_json_path)
    return model


def generate_model(model_file, loss_function="weighted",
                   metrics=minh_dice_coef_metric,
                   initial_learning_rate=0.001,
                   weight_tv_to_main_loss=0.1,
                   labels=[1, 2, 4]):

    model = load_model_multi_gpu(model_file)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         initial_learning_rate=initial_learning_rate,
                         alpha=weight_tv_to_main_loss,
                         labels=labels)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    name_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(name_gpus)


def compile_model(model, loss_function="weighted",
                  labels=[1, 2],
                  metrics=None,
                  initial_learning_rate=0.001,
                  alpha=0.00001):
    try:
        num_gpus = get_available_gpus()
        model = multi_gpu_model(model, gpus=num_gpus)
        print('!! train on multi gpus')
    except:
        print('!! train on single gpu')
        pass
    if loss_function == "tversky":
        loss = tversky_loss
    elif loss_function == "minh":
        loss = minh_dice_coef_loss
    elif loss_function == "tv_minh":
        loss = tv_minh_loss(alpha=alpha)
    elif loss_function == "tv_weighted":
        loss = tv_weighted_loss(alpha=alpha)
    elif loss_function == "weighted":
        # loss = weighted_dice_coefficient_loss(labels=labels)
        loss = weighted_dice_coefficient_loss
    elif loss_function == "categorical":
        loss = categorical_crossentropy
    if loss_function == "casweighted":
        model.compile(optimizer=Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999),
                      loss={'out_whole': dice_coefficient_loss,
                            'out_core': dice_coefficient_loss
                            },
                      loss_weights={'out_whole': 1,
                                    'out_core': 1
                                    }
                      )
    elif loss_function == "sepweighted":
        model.compile(optimizer=Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999),
                      loss={'out_1': dice_coefficient_loss,
                            'out_2': dice_coefficient_loss
                            },
                      loss_weights={'out_1': 1,
                                    'out_2': 1
                                    }
                      )
    else:
        model.compile(optimizer=Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999),
                      loss=loss)

    return model


def main():
    model = casnet_v10(input_shape=(4, 160, 192, 128))
    model.summary()


if __name__ == "__main__":
    main()
