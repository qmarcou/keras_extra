from unittest import TestCase
from keras_utils import utils
import tensorflow as tf
import numpy as np


class Test_axes_funcs(tf.test.TestCase):
    def test_move_axis_to_last_dim(self):
        x = tf.ones(shape=(2, 3, 4))
        self.assertShapeEqual(tf.ones(shape=(3, 4, 2)),
                              utils.move_axis_to_last_dim(x, 0))
        self.assertShapeEqual(tf.ones(shape=(2, 4, 3)),
                              utils.move_axis_to_last_dim(x, 1))
        self.assertShapeEqual(x,
                              utils.move_axis_to_last_dim(x, 2))
        self.assertShapeEqual(tf.ones(shape=(2, 4, 3)),
                              utils.move_axis_to_last_dim(x, -2))
        self.assertShapeEqual(tf.ones(shape=(4, 2, 3)),
                              utils.move_axis_to_last_dim(x, [0, 1]))
        self.assertShapeEqual(tf.ones(shape=(4, 3, 2)),
                              utils.move_axis_to_last_dim(x, [1, 0]))
        self.assertShapeEqual(tf.ones(shape=(4, 3, 2)),
                              utils.move_axis_to_last_dim(x, [-2, 0]))
        x = tf.concat([tf.ones(shape=(1, 3, 4)),
                       tf.ones(shape=(1, 3, 4)) * 2], axis=0)
        exp_x = tf.concat([tf.ones(shape=(3, 4, 1)),
                           tf.ones(shape=(3, 4, 1)) * 2], axis=2)
        self.assertAllEqual(exp_x,
                            utils.move_axis_to_last_dim(x, 0))

    def test_move_axis_to(self):
        # Check shape changes as expected
        x = tf.ones(shape=(2, 3, 4))
        self.assertShapeEqual(tf.ones(shape=(3, 4, 2)),
                              utils.move_axis_to(x, 0, 2))
        self.assertShapeEqual(tf.ones(shape=(3, 4, 2)),
                              utils.move_axis_to(x, 0, -1))
        self.assertShapeEqual(tf.ones(shape=(3, 2, 4)),
                              utils.move_axis_to(x, 0, 1))
        self.assertShapeEqual(x,
                              utils.move_axis_to(x, 0, 0))
        self.assertShapeEqual(tf.ones(shape=(3, 2, 4)),
                              utils.move_axis_to(x, 1, 0))
        self.assertShapeEqual(tf.ones(shape=(4, 2, 3)),
                              utils.move_axis_to(x, -1, 0))
        # Check that the values are moved as expected
        x = tf.concat([tf.ones(shape=(1, 3, 4)),
                       tf.ones(shape=(1, 3, 4)) * 2], axis=0)
        exp_x = tf.concat([tf.ones(shape=(3, 4, 1)),
                           tf.ones(shape=(3, 4, 1)) * 2], axis=2)
        self.assertAllEqual(exp_x,
                            utils.move_axis_to(x, 0, -1))

    def test_shape_without_axis(self):
        x = tf.ones(shape=(2, 3, 4))
        self.assertAllEqual(tf.constant([3, 4]),
                            utils.shape_without_axis(x, 0))
        self.assertAllEqual(tf.constant([4]),
                            utils.shape_without_axis(x, [0, 1]))
        self.assertAllEqual(tf.constant([3]),
                            utils.shape_without_axis(x, [0, -1]))


class Test(tf.test.TestCase):
    def test_compute_rank(self):
        x = tf.constant([0, 1, 2, 3])
        self.assertAllEqual(np.array([1, 2, 3, 4]),
                            utils.compute_ranks(values=x,
                                                direction='ASCENDING'))
        self.assertAllEqual(np.array([4, 3, 2, 1]),
                            utils.compute_ranks(values=x,
                                                direction='DESCENDING'))
        x = [5, 3, 8, 4, 6, 2]
        self.assertAllEqual(np.array([4, 2, 6, 3, 5, 1]),
                            utils.compute_ranks(values=x,
                                                direction='ASCENDING'))
        self.assertAllEqual(np.array([3, 5, 1, 4, 2, 6]),
                            utils.compute_ranks(values=x,
                                                direction='DESCENDING'))
