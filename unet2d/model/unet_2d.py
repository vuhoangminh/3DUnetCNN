from keras import backend as K
from keras.engine import Input, Model
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Deconvolution2D, Dense, GlobalAveragePooling2D,
                          MaxPooling2D, Permute, PReLU, Reshape, UpSampling2D,
                          multiply)
from keras.layers.merge import concatenate

from unet3d.metrics import (dice_coefficient, dice_coefficient_loss,
                            get_label_dice_coefficient_function,
                            minh_dice_coef_loss, minh_dice_coef_metric,
                            soft_dice_loss, soft_dice_numpy, tv_minh_loss,
                            tversky_loss, weighted_dice_coefficient_loss)
from unet3d.utils.model_utils import compile_model

from unet2d.model.blocks import get_up_convolution2d
from unet2d.model.blocks import squeeze_excite_block2d
from unet2d.model.blocks import create_convolution_block2d


K.set_image_data_format("channels_first")


def unet_model_2d(input_shape, pool_size=(2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False,
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
    current_layer = inputs
    levels = list()

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


def main():

    model = unet_model_2d(input_shape=(1, 128, 128),
                          n_labels=3)
    model.summary()
    print("done")


if __name__ == "__main__":
    main()
