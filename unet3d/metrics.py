import numpy as np

from functools import partial

from keras import backend as K
from keras.losses import categorical_crossentropy

from brats.config import config

from unet3d.utils import metrics_utils as utils

K.set_image_data_format("channels_first")


def dice_coefficient(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient_old(y_true, y_pred, smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    ndim = len(y_true._keras_shape)
    if K.image_data_format() == "channels_first":
        axis = (-3, -2, -1) if ndim == 5 else (-2, -1)
    else:
        axis = (-4, -3, -2) if ndim == 5 else (-3, -2)

    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient(y_true, y_pred, labels=config["labels"], weights=[1, 1, 1]):
    distance = 0
    for label in range(len(labels)):
        dice_coef_class = dice_coefficient(y_true[:, label], y_pred[:, label])
        dice_coef_class_weighted = dice_coef_class*weights[label]
        distance = dice_coef_class_weighted + distance
    return distance/3


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def minh_dice_coef_loss_old(y_true, y_pred, labels=config["labels"]):
    distance = 0
    for label in range(len(labels)):
        dice_coef_class = dice_coefficient(y_true[:, label], y_pred[:, label])
        distance = 1 - dice_coef_class + distance
    return distance


def minh_dice_coef_metric_old(y_true, y_pred, labels=config["labels"]):
    return (len(labels)-minh_dice_coef_loss_old(y_true, y_pred, labels=labels))/len(labels)


def minh_dice_coef_loss(y_true, y_pred, labels=config["labels"], weights=[2, 1, 3]):
    distance = 0
    for label in range(len(labels)):
        dice_coef_class = dice_coefficient(y_true[:, label], y_pred[:, label])
        dice_coef_class_weighted = dice_coef_class*weights[label]/6
        distance = 1 - dice_coef_class_weighted + distance
    return distance


def minh_dice_coef_metric(y_true, y_pred, labels=config["labels"], weights=[2, 1, 3]):
    distance = 0
    for label in range(len(labels)):
        dice_coef_class = dice_coefficient(y_true[:, label], y_pred[:, label])
        dice_coef_class_weighted = dice_coef_class*weights[label]/6
        distance = dice_coef_class_weighted + distance
    return distance


def tversky(y_true, y_pred, smooth=0.00001):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.8
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


# focal loss with multi label
def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        # 1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(
            prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(
            target_tensor, zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * \
            tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        # 2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(
            prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [total_num / ff for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff/sum_ for ff in classes_w_t1]  # scale
        classes_w_tensor = tf.convert_to_tensor(
            classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(
            target_tensor, zeros), classes_weight, zeros)

        # 3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_sum(balanced_fl)

        # 4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        final_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(
            K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return final_loss
    return focal_loss_fixed


def ignore_unknown_xentropy(ytrue, ypred):
    return (1-ytrue[:, :, :, 0])*categorical_crossentropy(ytrue, ypred)


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score
    if K.image_data_format() == 'channels_last':
        axes = tuple(range(1, len(y_pred.shape)-1))
    else:
        axes = tuple(range(2, len(y_pred.shape)))

    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    # average over classes and batch
    return 1 - np.mean(numerator / (denominator + epsilon))


def soft_dice_numpy(y_pred, y_true, eps=1e-7):
    '''
    c is number of classes
    :param y_pred: b x c x X x Y( x Z...) network output, must sum to 1 over c channel (such as after softmax)
    :param y_true: b x c x X x Y( x Z...) one hot encoding of ground truth
    :param eps: 
    :return: 
    '''

    axes = tuple(range(2, len(y_pred.shape)))
    intersect = np.sum(y_pred * y_true, axes)
    denom = np.sum(y_pred + y_true, axes)
    return - (2. * intersect / (denom + eps)).mean()


def tv_ndim_score(x, beta=2):
    r"""Implements the N-dim version of function
    $$TV^{\beta}(x) = \sum_{whc} \left ( \left ( x(h, w+1, c) - x(h, w, c) \right )^{2} +
    \left ( x(h+1, w, c) - x(h, w, c) \right )^{2} \right )^{\frac{\beta}{2}}$$
    to return total variation for all images in the batch.
    """
    def normalize(input_tensor, output_tensor):
        """Normalizes the `output_tensor` with respect to `input_tensor` dimensions.
        This makes regularizer weight factor more or less uniform across various input image dimensions.
        Args:
            input_tensor: An tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
                    channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
            output_tensor: The tensor to normalize.
        Returns:
            The normalized tensor.
        """
        image_dims = utils.get_img_shape(input_tensor)[1:]
        return output_tensor / np.prod(image_dims)

    image_dims = K.ndim(x) - 2

    # Constructing slice [1:] + [:-1] * (image_dims - 1) and [:-1] * (image_dims)
    start_slice = [slice(1, None, None)] + [slice(None, -1, None)
                                            for _ in range(image_dims - 1)]
    end_slice = [slice(None, -1, None) for _ in range(image_dims)]
    samples_channels_slice = [slice(None, None, None), slice(None, None, None)]

    # Compute pixel diffs by rolling slices to the right per image dim.
    tv = None
    for i in range(image_dims):
        ss = tuple(samples_channels_slice + start_slice)
        es = tuple(samples_channels_slice + end_slice)
        diff_square = K.square(x[utils.slicer[ss]] - x[utils.slicer[es]])
        tv = diff_square if tv is None else tv + diff_square

        # Roll over to next image dim
        start_slice = np.roll(start_slice, 1).tolist()
        end_slice = np.roll(end_slice, 1).tolist()

    tv = K.sum(K.pow(tv, beta / 2.))
    return normalize(x, tv)


def tv_ndim_loss(x, beta=2):
    return tv_ndim_score(x, beta=beta)


def tv_minh_loss(alpha=0.1):
    def loss(y_true, y_pred):
        return alpha*tv_ndim_loss(y_pred) + minh_dice_coef_loss(y_true, y_pred)
    return loss


def tv_weighted_loss(alpha=0.1):
    def loss(y_true, y_pred):
        return alpha*tv_ndim_loss(y_pred) + weighted_dice_coefficient_loss(y_true, y_pred)
    return loss


def loss(out_whole, out_core, out_enh):

    loss_whole = 

    loss_L2 = mse(inp, out_VAE)

    loss_KL = (1 / n) * K.sum(
        K.square(z_mean) + z_var - K.log(z_var) - 1,
        axis=-1
    )

    def loss_(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
        loss_dice = (2. * intersection) / (
            K.sum(K.square(y_true_f), -1) + K.sum(K.square(y_pred_f), -1) + e)

        return loss_dice + 0.1 * loss_L2 + 0.1 * loss_KL

    return loss_


    ndim = len(y_true._keras_shape)
    if K.image_data_format() == "channels_first":
        axis = (-3, -2, -1) if ndim == 5 else (-2, -1)
    else:
        axis = (-4, -3, -2) if ndim == 5 else (-3, -2)

    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))



dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
