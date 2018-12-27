"""Train CRF and BiLSTM-CRF on CONLL2000 chunking data,
similar to https://arxiv.org/pdf/1508.01991v1.pdf.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy
from collections import Counter

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.datasets import conll2000

EPOCHS = 10
EMBED_DIM = 200
BiRNN_UNITS = 200


def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics,
    reports per classs recall, precision and F1 score'''
    y_true = numpy.asarray(y_true).ravel()
    y_pred = numpy.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('',
                                                    'recall',
                                                    'precision',
                                                    'f1-score',
                                                    'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total',
                    sum(report2[0]) / N,
                    sum(report2[1]) / N,
                    sum(report2[2]) / N, N) + '\n')


# ------
# Data
# -----

# conll200 has two different targets, here will only use
# IBO like chunking as an example
train, test, voc = conll2000.load_data()
(train_x, _, train_y) = train
(test_x, _, test_y) = test
(vocab, _, class_labels) = voc

# --------------
# 1. Regular CRF
# --------------

print('==== training CRF ====')

model = Sequential()
model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
crf = CRF(len(class_labels), sparse_target=True)
model.add(crf)
model.summary()


# -------------
# 2. BiLSTM-CRF
# -------------

print('==== training BiLSTM-CRF ====')

model = Sequential()
model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
crf = CRF(len(class_labels), sparse_target=True)
model.add(crf)
model.summary()


# -------------
# 3.   Unet-CRF
# -------------

from unet3d.model.unet_new import unet_model_3d
model = unet_model_3d(input_shape=(4,128,128,128), n_labels=3, is_crf=True)
model.summary()