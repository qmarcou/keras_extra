from unittest import TestCase
from keras_utils import stats
import tensorflow as tf
import numpy as np


class Test_NanPercentile(tf.test.TestCase):
    def test_nanpercentile(self):
        x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9],dtype=tf.float32)
        self.assertEqual(5, stats.nanpercentile(x, 50))
        self.assertRaises(ValueError,
                          stats.nanpercentile, x, 50, axis=0)
        x_nan = tf.concat([tf.fill(value=np.nan, dims=(4,)),
                           x,
                           tf.fill(value=np.nan, dims=(4,))], axis=0)
        p = [10.0, 10.5, 25.0, 50.0, 75.0]
        self.assertAllEqual(stats.nanpercentile(x, p),
                            stats.nanpercentile(x_nan, p))
