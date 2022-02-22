from unittest import TestCase

import keras

from keras_utils import losses
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import Reduction
from numpy.testing import assert_array_almost_equal
import numpy as np
import tensorflow as tf

class Test(TestCase):
    def test_weighted_binary_crossentropy(self):
        wbc = losses.weighted_binary_crossentropy
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]] # shape nsamples*noutput
        weights = [[1.0, 1.0], [1.0, 1.0]] # shape noutput*2
        assert_array_almost_equal(binary_crossentropy(y_true, y_pred,
                                                      from_logits=False),
                                  wbc(y_true, y_pred, weights,
                                      from_logits=False))
        # Add more weight to positive examples
        weights = [[1.0, 1.0], [1.0, 3.0]]
        assert_array_almost_equal(np.array([1.83258105, 0.713558]),
                                  wbc(y_true, y_pred, weights,
                                      from_logits=False))
        # Pass tensorflow tensors as input instead of lists
        assert_array_almost_equal(np.array([1.83258105, 0.713558]),
                                  wbc(tf.constant(y_true),
                                      tf.constant(y_pred),
                                      tf.constant(weights),
                                      from_logits=False))
        # Check the wrapper loss class behavior
        wbce = losses.WeightedBinaryCrossentropy(class_weights=weights,
                                                 from_logits=False,
                                                 reduction=Reduction.NONE)
        assert_array_almost_equal(np.array([1.83258105, 0.713558]),
                                  wbce(y_true, y_pred).numpy())
        # with SUM_OVER_BATCH_SIZE reduction (average over batch size)
        wbce = losses.WeightedBinaryCrossentropy(class_weights=weights,
                                                 from_logits=False,
                                                 reduction=Reduction.SUM_OVER_BATCH_SIZE)
        assert_array_almost_equal(np.array(1.273069525),
                                  wbce(y_true, y_pred).numpy())
        # Inside a model pipeline
        # check that the loss function handles correctly tensors with unknown
        # dimension (at model initialization batch size is unknown and shape
        # the first dimension is None and the passed vectors are placeholders
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile(loss=wbce)
        model.fit(x=[1, 2],
                  y=y_true)
