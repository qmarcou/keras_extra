from unittest import TestCase
import tensorflow as tf
from keras_utils import layers
import numpy as np


class Test(tf.test.TestCase):
    def test_extremum_constraint_module(self):
        adj_mat = np.array([[0, 0], [1, 0]])  # 1 is child of 0
        # Check constructor behavior
        self.assertRaises(NotImplementedError, layers.ExtremumConstraintModule,
                          activation="sigmoid",
                          extremum="min",
                          adjacency_matrix=adj_mat,
                          sparse_adjacency=True)
        ecm_layer = layers.ExtremumConstraintModule(activation="linear",
                                                    extremum="min",
                                                    adjacency_matrix=adj_mat,
                                                    sparse_adjacency=False)
        # sample size 1, parent class with smaller val
        input_logits = tf.constant(np.array([[1.0, 2.0]]), dtype=tf.float32)
        self.assertAllEqual(np.array([[1.0, 1.0]]),
                            ecm_layer(input_logits))
