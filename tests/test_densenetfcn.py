from unet3d.model import densefcn_model_3d
from unet3d.model import isensee2017_model
from unet3d.model import unet_model_3d
from keras.utils import plot_model
from unet3d.model.densenetfcn3d import densefcn_model_3d
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_contrib.applications import densenet
import sys
import os
import keras.backend as K

from keras.models import Model

import sys
sys.path.append('external/Fully-Connected-DenseNets-Semantic-Segmentation')


weight_decay = 1E-4
batch_momentum = 0.9
batch_shape = None
classes = 21
include_top = False
activation = 'sigmoid'
target_size = (320, 320)
input_shape = target_size + (3,)

# input_shape = _obtain_input_shape(input_shape,
#                                   default_size=32,
#                                   min_size=16,
#                                   data_format=K.image_data_format(),
#                                   include_top=False)

# K.set_image_data_format('channels_last')

# if batch_shape:
#     img_input = Input(batch_shape=batch_shape)
#     image_size = batch_shape[1:3]
# else:
#     img_input = Input(shape=input_shape)
#     image_size = input_shape[0:2]


# model = densenet.DenseNetFCN(input_shape)
# model.summary()
K.set_image_data_format('channels_first')


input_shape = (4, 64, 64, 64)
model = densefcn_model_3d(input_shape=input_shape,
                          classes=3,
                          nb_dense_block=5,
                          nb_layers_per_block=4,
                          early_transition=True,
                          dropout_rate=0.5)
model.summary()                          
plot_model(model, to_file='densenetfcn3d.png', show_shapes=True)

# from unet3d.model.densenetfcn2d import densefcn_model_2d
# input_shape = (4, 64, 64)
# model = densefcn_model_2d(input_shape=input_shape, classes=3)
# plot_model(model, to_file='densenetfcn2d.png', show_shapes=True)


# model = unet_model_3d(input_shape=input_shape,
#                       n_labels=3)
# plot_model(model, to_file='unet3d.png', show_shapes=True)

# model = isensee2017_model(input_shape=input_shape,
#                           n_labels=3)
# plot_model(model, to_file='isensee3d.png', show_shapes=True)
