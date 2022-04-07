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
        ecm_layer_min = layers.ExtremumConstraintModule(activation="linear",
                                                        extremum="min",
                                                        adjacency_matrix=adj_mat,
                                                        sparse_adjacency=False)
        ecm_layer_max = layers.ExtremumConstraintModule(activation="linear",
                                                        extremum="max",
                                                        adjacency_matrix=adj_mat,
                                                        sparse_adjacency=False)
        # sample size 1, 2 class 1 relationship
        input_logits = tf.constant(np.array([[-1.0, 2.0]]), dtype=tf.float32)
        self.assertAllEqual(np.array([[-1.0, -1.0]]),
                            ecm_layer_min(input_logits))
        self.assertAllEqual(np.array([[-1.0, 2.0]]),
                            ecm_layer_max(input_logits))
        # Test non linear activation function
        ecm_layer_min_sigm = layers.ExtremumConstraintModule(activation="sigmoid",
                                                        extremum="min",
                                                        adjacency_matrix=adj_mat,
                                                        sparse_adjacency=False)
        self.assertAllCloseAccordingToType(
            tf.keras.activations.sigmoid(np.array([[-1.0, -1.0]])),
            ecm_layer_min_sigm(input_logits))
        # sample size 2
        input_logits = tf.constant(np.array([[-1.0, 2.0],
                                             [2.0, -1.0]]), dtype=tf.float32)
        self.assertAllEqual(np.array([[-1.0, -1.0],
                                      [2.0, -1.0]]),
                            ecm_layer_min(input_logits))
        self.assertAllEqual(np.array([[-1.0, 2.0],
                                      [2.0, 2.0]]),
                            ecm_layer_max(input_logits))
        # 3 class adjacency matrix, test coherent classification cases
        # 1 and 2 are child of 0
        # 3 is child of 1 (hence of 0)
        adj_mat = np.array([[0, 0, 0, 0], [1, 0, 0, 0],
                            [1, 0, 0, 0], [1, 1, 0, 0]])
        input_logits = tf.constant(np.array([[4.0, 3.0, -1.0, 2.0],
                                             [4.0, 2.0, -1.0, 3.0],
                                             [-1.0, 2.0, -1.0, 3.0]]),
                                   dtype=tf.float32)
        ecm_layer_min = layers.ExtremumConstraintModule(activation="linear",
                                                        extremum="min",
                                                        adjacency_matrix=adj_mat,
                                                        sparse_adjacency=False)
        # transpose adj_mat for max to give a parent->child adj_mat
        ecm_layer_max = layers.ExtremumConstraintModule(activation="linear",
                                                        extremum="max",
                                                        adjacency_matrix=adj_mat
                                                        .transpose(),
                                                        sparse_adjacency=False)
        self.assertAllEqual(np.array([[4.0, 3.0, -1.0, 2.0],
                                      [4.0, 2.0, -1.0, 2.0],
                                      [-1.0, -1.0, -1.0, -1.0]]),
                            ecm_layer_min(input_logits))
        self.assertAllEqual(np.array([[4.0, 3.0, -1.0, 2.0],
                                      [4.0, 3.0, -1.0, 3.0],
                                      [3.0, 3.0, -1.0, 3.0]]),
                            ecm_layer_max(input_logits))

        # Test building with partially unknown input_shape
        ecm_layer_max.build(input_shape=tf.TensorShape([None, 4]))
