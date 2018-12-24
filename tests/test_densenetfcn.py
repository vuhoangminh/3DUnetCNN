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



from unet3d.model.densenetfcn3d import DenseNetFCN_3D
model = DenseNetFCN_3D((4,128,128,128))
model.summary()


# from unet3d.model.densenetfcn2d import DenseNetFCN
# model = DenseNetFCN((4,240,240))
# model.summary()