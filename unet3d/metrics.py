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


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    if K.image_data_format() == 'channels_last':
        compute_axis = tuple(range(1, len(y_pred.shape)-1))
    else:
        compute_axis = tuple(range(2, len(y_pred.shape)))
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=compute_axis) + smooth/2)/(K.sum(y_true,
                                                                    axis=compute_axis) + K.sum(y_pred,
                                                                                               axis=compute_axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def minh_dice_coef_loss(y_true, y_pred, labels=config["labels"]):
    distance = 0
    for label in range(len(labels)):
        dice_coef_class = dice_coefficient(y_true[:, label], y_pred[:, label])
        distance = 1 - dice_coef_class + distance
    return distance


def minh_dice_coef_metric(y_true, y_pred, labels=config["labels"]):
    return (len(labels)-minh_dice_coef_loss(y_true, y_pred, labels=labels))/len(labels)


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    if K.image_data_format() == 'channels_last':
        compute_axis = tuple(range(1, len(y_pred.shape)-1))
    else:
        compute_axis = tuple(range(2, len(y_pred.shape)))

    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true

    num = K.sum(p0*g0, compute_axis)
    den = num + alpha*K.sum(p0*g1, compute_axis) + \
        beta*K.sum(p1*g0, compute_axis)

    # when summing over classes, T has dynamic range [0 Ncl]
    T = K.sum(num/den)

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output),
                  axis=-1)


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


def tv_ndim_loss(x, beta=2):
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


def tv_minh_loss(y_true, y_pred, labels=config["labels"]):
    return tv_ndim_loss(y_pred) + minh_dice_coef_loss(y_true, y_pred, labels=config["labels"])


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
