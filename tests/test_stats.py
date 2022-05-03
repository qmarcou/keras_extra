from unittest import TestCase
from keras_utils import stats
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Test_NanPercentile(tf.test.TestCase):
    def test_nanpercentile(self):
        # Test 1D case
        x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.float32)
        self.assertEqual(5, stats.nanpercentile(x, 50))
        self.assertEqual(5, stats.nanpercentile(x, 50, axis=0))

        x_nan = tf.concat([tf.fill(value=np.nan, dims=(4,)),
                           x,
                           tf.fill(value=np.nan, dims=(4,))], axis=0)
        p = [10.0, 10.5, 25.0, 50.0, 75.0]
        self.assertAllEqual(stats.nanpercentile(x, p),
                            stats.nanpercentile(x_nan, p))

        # Test cases of only NaNs and
        # correct shapes returned by _percentile_wrapper
        x = [[np.nan] * 9,
             [np.nan] * 9]
        self.assertEqual(
            True,
            tf.math.is_nan(stats.nanpercentile(x,
                                               q=50, axis=None)))
        self.assertAllEqual(
            np.array([True, True]),
            tf.math.is_nan(stats.nanpercentile(x,
                                               q=[50, 66.6], axis=None)))
        x = [[1, 2, 3, 4, 5, 6, 7, 8, 9],
             [np.nan] * 9]
        self.assertAllEqual(
            np.array([[False, True],
                      [False, True]]),
            tf.math.is_nan(
                stats.nanpercentile(x,
                                    q=[50, 66.6], axis=1)))

        # Check that without NaNs the function returns the same results
        # as tfp.stats.percentile
        rand_x = tf.random.uniform(shape=(2, 3, 4, 5))
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

        # Check NaN stability
        rand_x_nan = tf.concat([tf.fill(value=np.nan, dims=(2, 3, 1, 5)),
                                rand_x,
                                tf.fill(value=np.nan, dims=(2, 3, 1, 5))],
                               axis=2)
        self.assertAllEqual(
            stats.nanpercentile(x=rand_x, q=[50, 75], axis=2),
            stats.nanpercentile(x=rand_x_nan, q=[50, 75], axis=2)
        )
        self.assertAllEqual(
            stats.nanpercentile(x=rand_x, q=[50, 75], axis=[1, 2, 3]),
            stats.nanpercentile(x=rand_x_nan, q=[50, 75], axis=[1, 2, 3])
        )
        self.assertAllEqual(
            stats.nanpercentile(x=rand_x, q=[50, 75], axis=None),
            stats.nanpercentile(x=rand_x_nan, q=[50, 75], axis=None)
        )

        # Test error handling for invalid arguments
        self.assertRaises(tf.errors.InvalidArgumentError,
                          stats.nanpercentile,
                          x=rand_x, q=[[50, 75], [20, 80]], axis=None)

        self.assertRaises(tf.errors.InvalidArgumentError,
                          stats.nanpercentile,
                          x=rand_x, q=[[50, 75], [20, 80]], axis=-1)

        self.assertRaises(tf.errors.InvalidArgumentError,
                          stats.nanpercentile,
                          x=rand_x, q=[50, 75], axis=[[-1], [-2]])
