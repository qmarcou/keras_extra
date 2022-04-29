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
