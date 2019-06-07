import numpy as np
from keras import backend as K
import keras.layers as KL
from keras.layers import (Activation, BatchNormalization, Conv3D,
                          Deconvolution3D, Dense, GlobalAveragePooling3D,
                          MaxPooling3D, Permute, PReLU, Reshape, UpSampling3D,
                          multiply, Add)
from keras import regularizers
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras.utils.generic_utils import get_custom_objects


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


get_custom_objects().update({'BatchNorm': BatchNorm})


class GroupNormalization(Layer):
    """Group normalization layer

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes

    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv3D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'GroupNormalization': GroupNormalization})


def create_convolution_block3d(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                               padding='same', strides=(1, 1, 1), instance_normalization=False,
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
    layer = Conv3D(n_filters, kernel,
                   padding=padding,
                   strides=strides,
                   kernel_regularizer=regularizers.l2(l=weight_decay))(input_layer)
    if batch_normalization:
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
        layer = squeeze_excite_block3d(layer)
    return layer


def compute_level_output_shape3d(n_filters, depth, pool_size, image_shape):
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


def get_up_convolution3d(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                         deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)


def squeeze_excite_block3d(input, ratio=16):
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


def conv_block_resnet3d(input_layer, kernel_size, n_filters, stage, block,
                        use_bias=True, train_bn=True,
                        padding='same', strides=(1, 1, 1),
                        weight_decay=1e-5):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_layer: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        n_filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = n_filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(nb_filter1, (1, 1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=use_bias,
               padding=padding,
               kernel_regularizer=regularizers.l2(l=weight_decay))(input_layer)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size),
               name=conv_name_base + '2b', use_bias=use_bias,
               padding=padding,
               kernel_regularizer=regularizers.l2(l=weight_decay))(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv3D(nb_filter3, (1, 1, 1), name=conv_name_base +
               '2c', use_bias=use_bias,
               padding=padding,
               kernel_regularizer=regularizers.l2(l=weight_decay))(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = Conv3D(nb_filter3, (1, 1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=use_bias,
                      padding=padding,
                      kernel_regularizer=regularizers.l2(l=weight_decay))(input_layer)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


# based on this paper: https://arxiv.org/pdf/1603.05027.pdf
def conv_block_resnet3d_proposed(input_layer, kernel_size, n_filters, stage, block,
                                 use_bias=True, train_bn=True,
                                 padding='same', strides=(1, 1, 1),
                                 weight_decay=1e-5):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_layer: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        n_filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = n_filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNorm(name=bn_name_base + '2a')(input_layer, training=train_bn)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter1, (kernel_size, kernel_size, kernel_size),
               strides=strides,
               name=conv_name_base + '2a', use_bias=use_bias,
               padding=padding,
               kernel_regularizer=regularizers.l2(l=weight_decay))(x)

    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size),
               strides=strides,
               name=conv_name_base + '2b', use_bias=use_bias,
               padding=padding,
               kernel_regularizer=regularizers.l2(l=weight_decay))(x)

    x = Add()([x, input_layer])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def get_down_sampling(input_layer, pool_size=(2, 2, 2), depth=4):
    levels = list()
    current_layer = input_layer
    for layer_depth in range(depth):
        if layer_depth == 0:
            levels.append(current_layer)
        else:
            current_layer = MaxPooling3D(pool_size=pool_size)(current_layer)
            levels.append(current_layer)
    return levels
