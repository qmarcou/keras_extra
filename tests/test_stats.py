from unittest import TestCase
from keras_utils import stats
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Test_NanPercentile(tf.test.TestCase):
    def test_nanpercentile(self):
        x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.float32)
        self.assertEqual(5, stats.nanpercentile(x, 50))
        self.assertEqual(5, stats.nanpercentile(x, 50, axis=0))

        x_nan = tf.concat([tf.fill(value=np.nan, dims=(4,)),
                           x,
                           tf.fill(value=np.nan, dims=(4,))], axis=0)
        p = [10.0, 10.5, 25.0, 50.0, 75.0]
        self.assertAllEqual(stats.nanpercentile(x, p),
                            stats.nanpercentile(x_nan, p))

        rand_x = tf.random.uniform(shape=(2, 3, 4, 5))
        # Check that without NaNs the function returns the same results
        # as tfp.stats.percentile
        self.assertAllEqual(
            tfp.stats.percentile(x=rand_x, q=50, axis=-1),
            stats.nanpercentile(x=rand_x, q=50, axis=-1)
        )
        self.assertAllEqual(
            tfp.stats.percentile(x=rand_x, q=[50], axis=-1),
            stats.nanpercentile(x=rand_x, q=[50], axis=-1)
        )
        self.assertAllEqual(
            tfp.stats.percentile(x=rand_x, q=[50, 75], axis=-1),
            stats.nanpercentile(x=rand_x, q=[50, 75], axis=-1)
        )
        self.assertAllEqual(
            tfp.stats.percentile(x=rand_x, q=[50, 75], axis=2),
            stats.nanpercentile(x=rand_x, q=[50, 75], axis=2)
        )
