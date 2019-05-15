import os
import numpy as np
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.applications import imagenet_utils


from keras.engine.topology import Layer as Layer_inh
from keras.layers.pooling import _Pooling2D, _Pooling3D

import six.moves.builtins as builtins
import keras.backend as K
from keras.layers import Layer


# def m_maxpool2d(x, pool_size, strides=(1, 1), border_mode='valid',
#                 dim_ordering='th'):
#     if border_mode == 'same':
#         w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
#         h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
#         padding = (w_pad, h_pad)
#     elif border_mode == 'valid':
#         padding = (0, 0)
#     else:
#         raise Exception('Invalid border mode: ' + str(border_mode))

#     if dim_ordering not in {'th', 'tf'}:
#         raise Exception('Unknown dim_ordering ' + str(dim_ordering))

#     if dim_ordering == 'tf':
#         x = x.dimshuffle((0, 3, 1, 2))

#     pool_out, ind = m_maxpool_2d_op(x, ds=pool_size, st=strides,
#                                     ignore_border=True,
#                                     padding=padding, )

#     if border_mode == 'same':
#         expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
#         expected_height = (x.shape[3] + strides[1] - 1) // strides[1]

#         pool_out = pool_out[:, :, :expected_width, :expected_height]

#     if dim_ordering == 'tf':
#         pool_out = pool_out.dimshuffle((0, 2, 3, 1))

#     return pool_out, ind


# def m_maxpool_2d_op(input, ds, ignore_border=None, st=None, padding=(0, 0)):
#     """Downscale the input by a specified factor
#     Takes as input a N-D tensor, where N >= 2. It downscales the input image by
#     the specified factor, by keeping only the maximum value of non-overlapping
#     patches of size (ds[0],ds[1])
#     Parameters
#     ----------
#     input : N-D theano tensor of input images
#         Input images. Max pooling will be done over the 2 last dimensions.
#     ds : tuple of length 2
#         Factor by which to downscale (vertical ds, horizontal ds).
#         (2,2) will halve the image in each dimension.
#     ignore_border : bool (default None, will print a warning and set to False)
#         When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
#         (3,3) otherwise.
#     st : tuple of two ints
#         Stride size, which is the number of shifts over rows/cols to get the
#         next pool region. If st is None, it is considered equal to ds
#         (no overlap on pooling regions).
#     padding : tuple of two ints
#         (pad_h, pad_w), pad zeros to extend beyond four borders of the
#         images, pad_h is the size of the top and bottom margins, and
#         pad_w is the size of the left and right margins.
#     """
#     if input.ndim < 2:
#         raise NotImplementedError('pool_2d requires a dimension >= 2')
#     if ignore_border is None:
#         warnings.warn(
#             "pool_2d() will have the parameter ignore_border"
#             " default value changed to True (currently"
#             " False). To have consistent behavior with all Theano"
#             " version, explicitly add the parameter ignore_border=True."
#             " On the GPU, using ignore_border=True is needed to use cuDNN."
#             " When using ignore_border=False and not using cuDNN, the only"
#             " GPU combination supported is when"
#             " `ds == st and padding == (0, 0) and mode == 'max'`."
#             " Otherwise, the convolution will be executed on CPU.",
#             stacklevel=2)
#         ignore_border = False
#     if input.ndim == 4:
#         op = MPool(ds, ignore_border, st=st, padding=padding)
#         output, ind = op(input)
#         return output, ind

#     # extract image dimensions
#     img_shape = input.shape[-2:]

#     # count the number of "leading" dimensions, store as dmatrix
#     batch_size = tensor.prod(input.shape[:-2])
#     batch_size = tensor.shape_padright(batch_size, 1)

#     # store as 4D tensor with shape: (batch_size,1,height,width)
#     new_shape = tensor.cast(tensor.join(0, batch_size,
#                                         tensor.as_tensor([1]),
#                                         img_shape), 'int64')
#     input_4D = tensor.reshape(input, new_shape, ndim=4)

#     # downsample mini-batch of images
#     op = MPool(ds, ignore_border, st=st, padding=padding)
#     output, ind = op(input_4D)

#     # restore to original shape
#     outshp = tensor.join(0, input.shape[:-2], output.shape[-2:])
#     output = tensor.reshape(output, outshp, ndim=input.ndim)
#     ind = tensor.reshape(ind, input.shape)
#     return output, ind


# class MPool(Pool):
#     """
#     For N-dimensional tensors, consider that the last two dimensions span
#     images. This Op downsamples these images by taking the max, sum or average
#     over different patch.
#     The constructor takes the max, sum or average or different input patches.
#     Parameters
#     ----------
#     ds : list or tuple of two ints
#         Downsample factor over rows and column.
#         ds indicates the pool region size.
#     ignore_border : bool
#         If ds doesnt' divide imgshape, do we include an extra row/col
#         of partial downsampling (False) or ignore it (True).
#     st : list or tuple of two ints or None
#         Stride size, which is the number of shifts over rows/cols to get the
#         next pool region. If st is None, it is considered equal to ds
#         (no overlap on pooling regions).
#     padding: tuple of two ints
#         (pad_h, pad_w), pad zeros to extend beyond four borders of the images,
#         pad_h is the size of the top and bottom margins, and pad_w is the size
#         of the left and right margins.
#     """

#     __props__ = ('ds', 'ignore_border', 'st', 'padding')

#     def __init__(self, ds, ignore_border, st, padding):
#         self.mode = 'max'
#         self.openmp = False
#         self.ds = ds
#         self.ignore_border = ignore_border
#         self.st = st
#         self.padding = padding

#     def make_node(self, x):
#         if x.type.ndim != 4:
#             raise TypeError()
#         x = tensor.as_tensor_variable(x)
#         # If the input shape are broadcastable we can have 0 in the output shape
#         broad = x.broadcastable[:2] + (False, False)
#         out1 = tensor.TensorType(x.dtype, broad)
#         out2 = tensor.TensorType(x.dtype, broad)
#         return gof.Apply(self, [x], [out1(), out2()])

#     def perform(self, node, inp, out):
#         x, = inp
#         z, ind = out
#         ind = numpy.zeros_like(x)
#         if len(x.shape) != 4:
#             raise NotImplementedError('Pool requires 4D input for now')
#         z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st,
#                                  self.padding)
#         if (z[0] is None) or (z[0].shape != z_shape):
#             z[0] = numpy.empty(z_shape, dtype=x.dtype)
#         zz = z[0]
#         # number of pooling output rows
#         pr = zz.shape[-2]
#         # number of pooling output cols
#         pc = zz.shape[-1]
#         ds0, ds1 = self.ds
#         st0, st1 = self.st
#         pad_h = self.padding[0]
#         pad_w = self.padding[1]
#         img_rows = x.shape[-2] + 2 * pad_h
#         img_cols = x.shape[-1] + 2 * pad_w
#         inc_pad = 0

#         # pad the image
#         if self.padding != (0, 0):
#             y = numpy.zeros(
#                 (x.shape[0], x.shape[1], img_rows, img_cols),
#                 dtype=x.dtype)
#             y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
#         else:
#             y = x

#         for n in xrange(x.shape[0]):
#             for k in xrange(x.shape[1]):
#                 for r in xrange(pr):
#                     row_st = r * st0
#                     row_end = builtins.min(row_st + ds0, img_rows)
#                     if not inc_pad:
#                         row_st = builtins.max(row_st, self.padding[0])
#                         row_end = builtins.min(row_end, x.shape[-2] + pad_h)
#                     for c in xrange(pc):
#                         col_st = c * st1
#                         col_end = builtins.min(col_st + ds1, img_cols)
#                         if not inc_pad:
#                             col_st = builtins.max(col_st, self.padding[1])
#                             col_end = builtins.min(col_end,
#                                                    x.shape[-1] + pad_w)
#                         cur_max = y[n, k, row_st, col_st]
#                         max_r, max_c = row_st, col_st
#                         for rr in xrange(row_st, row_end):
#                             for cc in xrange(col_st, col_end):
#                                 if y[n, k, rr, cc] > cur_max:
#                                     cur_max = y[n, k, rr, cc]
#                                     max_r, max_c = rr, cc
#                         zz[n, k, r, c] = cur_max
#                         ind[n, k, max_r, max_c] = 1

#     def infer_shape(self, node, in_shapes):
#         shp = self.out_shape(in_shapes[0], self.ds,
#                              self.ignore_border, self.st, self.padding)
#         return [shp, in_shapes[0]]

#     def grad(self, inp, grads):
#         x, = inp
#         gz, _ = grads
#         maxout, _ = self(x)
#         return [MaxPoolGrad(self.ds,
#                             ignore_border=self.ignore_border,
#                             st=self.st, padding=self.padding)(x, maxout, gz)]

#     def c_code(self, node, name, inp, out, sub):
#         x, = inp
#         z, ind = out

#         fail = sub['fail']
#         ignore_border = int(self.ignore_border)
#         ds0, ds1 = self.ds
#         st0, st1 = self.st
#         pd0, pd1 = self.padding
#         ccode = """
#         int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
#         int z_r, z_c; // shape of the output
#         int r, c; // shape of the padded_input
#         if(PyArray_NDIM(%(x)s)!=4)
#         {
#             PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
#             %(fail)s;
#         }
#         r = PyArray_DIMS(%(x)s)[2];
#         c = PyArray_DIMS(%(x)s)[3];
#         r += %(pd0)s * 2;
#         c += %(pd1)s * 2;
#         if (%(pd0)s != 0 && %(pd1)s != 0 && !%(ignore_border)s)
#             {
#               PyErr_SetString(PyExc_ValueError,
#                 "padding must be (0,0) when ignore border is False");
#               %(fail)s;
#             }
#         if (%(ignore_border)s)
#         {
#             // '/' in C is different from '/' in python
#             if (r - %(ds0)s < 0)
#             {
#               z_r = 0;
#             }
#             else
#             {
#               z_r = (r - %(ds0)s) / %(st0)s + 1;
#             }
#             if (c - %(ds1)s < 0)
#             {
#               z_c = 0;
#             }
#             else
#             {
#               z_c = (c - %(ds1)s) / %(st1)s + 1;
#             }
#         }
#         else
#         {
#             // decide how many rows the output has
#             if (%(st0)s >= %(ds0)s)
#             {
#                 z_r = (r - 1) / %(st0)s + 1;
#             }
#             else
#             {
#                 z_r = std::max(0, (r - 1 - %(ds0)s) / %(st0)s + 1) + 1;
#             }
#             // decide how many columns the output has
#             if (%(st1)s >= %(ds1)s)
#             {
#                 z_c = (c - 1) / %(st1)s + 1;
#             }
#             else
#             {
#                 z_c = std::max(0, (c - 1 - %(ds1)s) / %(st1)s + 1) + 1;
#             }
#         }
#         // memory allocation of z if necessary
#         if ((!%(z)s)
#           || *PyArray_DIMS(%(z)s)!=4
#           ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
#           ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
#           ||(PyArray_DIMS(%(z)s)[2] != z_r)
#           ||(PyArray_DIMS(%(z)s)[3] != z_c)
#           )
#         {
#           if (%(z)s) Py_XDECREF(%(z)s);
#           npy_intp dims[4] = {0,0,0,0};
#           dims[0]=PyArray_DIMS(%(x)s)[0];
#           dims[1]=PyArray_DIMS(%(x)s)[1];
#           dims[2]=z_r;
#           dims[3]=z_c;
#           //TODO: zeros not necessary
#           %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
#         }
#         // memory allocation of ind if necessary
#         if ((!%(ind)s)
#           || *PyArray_DIMS(%(ind)s)!=4
#           ||(PyArray_DIMS(%(ind)s)[0] != PyArray_DIMS(%(x)s)[0])
#           ||(PyArray_DIMS(%(ind)s)[1] != PyArray_DIMS(%(x)s)[1])
#           ||(PyArray_DIMS(%(ind)s)[2] != PyArray_DIMS(%(x)s)[2])
#           ||(PyArray_DIMS(%(ind)s)[3] != PyArray_DIMS(%(x)s)[3])
#           )
#         {
#           if (%(ind)s) Py_XDECREF(%(ind)s);
#           npy_intp dims[4] = {0,0,0,0};
#           dims[0]=PyArray_DIMS(%(x)s)[0];
#           dims[1]=PyArray_DIMS(%(x)s)[1];
#           dims[2]=PyArray_DIMS(%(x)s)[2];
#           dims[3]=PyArray_DIMS(%(x)s)[3];
#           //TODO: zeros not necessary
#           %(ind)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
#         }
#         // used for indexing a pool region inside the input
#         int r_st, r_end, c_st, c_end;
#         dtype_%(x)s collector; // temp var for the value in a region
#         if (z_r && z_c)
#         {
#             for(int b=0; b<PyArray_DIMS(%(x)s)[0]; b++){
#               for(int k=0; k<PyArray_DIMS(%(x)s)[1]; k++){
#                 int* index_x = new int [z_c];
#                 int* index_y = new int [z_r];
#                 int count = 0;
#                 for(int i=0; i< z_r; i++){
#                   r_st = i * %(st0)s;
#                   r_end = r_st + %(ds0)s;
#                   // skip the padding
#                   r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
#                   r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
#                   // from padded_img space to img space
#                   r_st -= %(pd0)s;
#                   r_end -= %(pd0)s;
#                   // handle the case where no padding, ignore border is True
#                   if (%(ignore_border)s)
#                   {
#                     r_end = r_end > r ? r : r_end;
#                   }
#                   for(int j=0; j<z_c; j++){
#                     c_st = j * %(st1)s;
#                     c_end = c_st + %(ds1)s;
#                     // skip the padding
#                     c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
#                     c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
#                     dtype_%(z)s * z = (
#                           (dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j)));
#                     // change coordinates from padding_img space into img space
#                     c_st -= %(pd1)s;
#                     c_end -= %(pd1)s;
#                     // handle the case where no padding, ignore border is True
#                     if (%(ignore_border)s)
#                     {
#                       c_end = c_end > c ? c : c_end;
#                     }
#                     // use the first element as the initial value of collector
#                     collector = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,r_st,c_st)))[0];
#                     int r_max = r_st, c_max = c_st;
#                     // go through the pooled region in the unpadded input
#                     for(int m=r_st; m<r_end; m++)
#                     {
#                       for(int n=c_st; n<c_end; n++)
#                       {
#                         dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
#                         if (a > collector) {
#                             collector = a;
#                             r_max = m;
#                             c_max = n;
#                         }
#                       }
#                     }
#                     //std::cout << count << ": " << r_max << ' ' << c_max << " | " << cur_max_ind << std::endl;
#                     count += 1;
#                     z[0] = collector;
#                     dtype_%(ind)s * ind = (
#                           (dtype_%(ind)s*)(PyArray_GETPTR4(%(ind)s, b, k, r_max, c_max)));
#                     ind[0] = 1;
#                   }
#                 }
#               }
#             }
#         }
#         """
#         return ccode % locals()


# #####################################################################################################

# def m_maxpool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
#                 dim_ordering='th'):
#     if border_mode == 'same':
#         # TODO: add implementation for border_mode="same"
#         raise Exception('border_mode="same" not supported with Theano.')
#     elif border_mode == 'valid':
#         ignore_border = True
#         padding = (0, 0)
#     else:
#         raise Exception('Invalid border mode: ' + str(border_mode))

#     if dim_ordering not in {'th', 'tf'}:
#         raise Exception('Unknown dim_ordering ' + str(dim_ordering))

#     if dim_ordering == 'tf':
#         x = x.dimshuffle((0, 4, 1, 2, 3))

#     # TODO: check dimensions manipulations
#     # pooling over conv_dim2, conv_dim1 (last two channels)
#     out_shape = x.shape
#     output, ind1 = m_maxpool_2d_op(input=x,
#                                    ds=(pool_size[1], pool_size[2]),
#                                    st=(strides[1], strides[2]),
#                                    ignore_border=ignore_border,
#                                    padding=padding)

#     # pooling over conv_dim3
#     pool_out, ind2 = m_maxpool_2d_op(input=output.dimshuffle(0, 1, 4, 3, 2),
#                                      ds=(1, pool_size[0]),
#                                      st=(1, strides[0]),
#                                      ignore_border=ignore_border,
#                                      padding=padding)

#     pool_out = pool_out.dimshuffle(0, 1, 4, 3, 2)
#     ind2 = ind2.dimshuffle(0, 1, 4, 3, 2)

#     ind2 = K.resize_volumes(ind2, 1, pool_size[1], pool_size[2], dim_ordering)
#     padded_ind2 = T.zeros(out_shape)
#     padded_ind2 = T.set_subtensor(
#         padded_ind2[:, :, :ind2.shape[2], :ind2.shape[3], :ind2.shape[4]], ind2)
#     ind = padded_ind2 * ind1
#     if dim_ordering == 'tf':
#         pool_out = pool_out.dimshuffle((0, 2, 3, 4, 1))

#     return pool_out, ind


# class MemoMaxPooling2D(_Pooling2D):
#     def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
#                  dim_ordering=K.image_dim_ordering(), **kwargs):
#         super(MemoMaxPooling2D, self).__init__(pool_size, strides, border_mode,
#                                                dim_ordering, **kwargs)

#     def _pooling_function(self, inputs, pool_size, strides,
#                           border_mode, dim_ordering):
#         output, ind = m_maxpool2d(inputs, pool_size, strides,
#                                   border_mode, dim_ordering)
#         self.ind = ind
#         return output


# class MemoUpSampling2D(Layer_inh):
#     def __init__(self, pooling_layer, size=(2, 2), shape=None, dim_ordering=K.image_dim_ordering(), **kwargs):
#         self.size = tuple(size)
#         assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
#         self.dim_ordering = dim_ordering
#         # self.input_spec = [InputSpec(ndim=4)]
#         self.ind = pooling_layer.ind
#         assert shape is None or len(shape) == 2, 'shape should be 2D vector'
#         self.shape = shape or pooling_layer.ind.shape[-2:]
#         super(MemoUpSampling2D, self).__init__(**kwargs)

#     def get_output_shape_for(self, input_shape):
#         if self.dim_ordering == 'th':
#             self.shape = self.shape or self.size * input_shape[:-2]
#             return (input_shape[0],
#                     input_shape[1],
#                     self.shape[0],
#                     self.shape[1])
#         elif self.dim_ordering == 'tf':
#             self.shape = self.shape or self.size * input_shape[1:-1]
#             return (input_shape[0],
#                     self.shape[0],
#                     self.shape[1],
#                     input_shape[3])
#         else:
#             raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

#     def call(self, x, mask=None):
#         # TODO: implement for dim_ordering='tf'
#         img = K.resize_images(x, self.size[0], self.size[1], self.dim_ordering)
#         padded = T.zeros(
#             (img.shape[0], img.shape[1], self.shape[0], self.shape[1]))
#         padded = T.set_subtensor(
#             padded[:, :, :img.shape[2], :img.shape[3]], img)
#         return T.switch(self.ind, padded, T.zeros_like(padded))

#     def get_config(self):
#         config = {'size': self.size}
#         base_config = super(MemoUpSampling2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

# #####################################################################################################


# class MemoMaxPooling3D(_Pooling3D):
#     def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode='valid',
#                  dim_ordering=K.image_dim_ordering(), **kwargs):
#         super(MemoMaxPooling3D, self).__init__(pool_size, strides, border_mode,
#                                                dim_ordering, **kwargs)

#     def _pooling_function(self, inputs, pool_size, strides,
#                           border_mode, dim_ordering):
#         output, ind = m_maxpool3d(
#             inputs, pool_size, strides, border_mode, dim_ordering)
#         self.ind = ind
#         return output


# class MemoUpSampling3D(Layer_inh):
#     def __init__(self, pooling_layer, size=(2, 2, 2), shape=None, dim_ordering=K.image_dim_ordering(), **kwargs):
#         self.size = tuple(size)
#         assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
#         self.dim_ordering = dim_ordering
#         # self.input_spec = [InputSpec(ndim=5)]
#         self.ind = pooling_layer.ind
#         assert shape is None or len(shape) == 3, 'shape should be 3D vector'
#         self.shape = shape or pooling_layer.ind.shape[-3:]
#         super(MemoUpSampling3D, self).__init__(**kwargs)

#     def get_output_shape_for(self, input_shape):
#         if self.dim_ordering == 'th':
#             return (input_shape[0],
#                     input_shape[1],
#                     self.shape[0],
#                     self.shape[1],
#                     self.shape[2])
#         elif self.dim_ordering == 'tf':
#             return (input_shape[0],
#                     self.shape[0],
#                     self.shape[1],
#                     self.shape[2],
#                     input_shape[4])
#         else:
#             raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

#     def call(self, x, mask=None):
#         # TODO: implement for dim_ordering='tf'
#         img = K.resize_volumes(
#             x, self.size[0], self.size[1], self.size[2], self.dim_ordering)
#         padded = T.zeros(
#             (img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.shape[2]))
#         padded = T.set_subtensor(
#             padded[:, :, :img.shape[2], :img.shape[3], :img.shape[4]], img)
#         return T.switch(self.ind, padded, T.zeros_like(padded))

#     def get_config(self):
#         config = {'size': self.size}
#         base_config = super(MemoUpSampling3D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim//ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1]*self.size[0],
                    input_shape[2]*self.size[1],
                    input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1]*self.size[0],
            mask_shape[2]*self.size[1],
            mask_shape[3]
        )


def segnet2d(input_shape,
             n_labels,
             kernel=3,
             pool_size=(2, 2),
             output_mode="sigmoid"):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape(
        (input_shape[0]*input_shape[1], n_labels),
        input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet2D")

    return model
