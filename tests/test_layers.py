from unittest import TestCase

import keras
import tensorflow as tf
from keras_utils import layers
import numpy as np
from scipy import sparse


class Test(tf.test.TestCase):
    def test_extremum_constraint_module(self):
        # Check constructor behavior
        for is_sparse_mat in [False, True]:
            adj_mat = np.array([[0, 0], [1, 0]])  # 1 is child of 0
            # if is_sparse_mat:
            #     adj_mat = sparse.coo_matrix(adj_mat)
            ecm_layer_min = layers.ExtremumConstraintModule(
                activation="linear",
                extremum="min",
                adjacency_matrix=adj_mat,
                sparse_adjacency=is_sparse_mat)
            ecm_layer_max = layers.ExtremumConstraintModule(
                activation="linear",
                extremum="max",
                adjacency_matrix=adj_mat,
                sparse_adjacency=is_sparse_mat)

            # sample size 1, 2 class 1 relationship
            input_logits = tf.constant(np.array([[-1.0, 2.0]]),
                                       dtype=tf.float32)
            self.assertAllEqual(np.array([[-1.0, -1.0]]),
                                ecm_layer_min(input_logits))
            self.assertAllEqual(np.array([[-1.0, 2.0]]),
                                ecm_layer_max(input_logits))
            # Test non linear activation function
            ecm_layer_min_sigm = layers.ExtremumConstraintModule(
                activation="sigmoid",
                extremum="min",
                adjacency_matrix=adj_mat,
                sparse_adjacency=is_sparse_mat)
            self.assertAllCloseAccordingToType(
                tf.keras.activations.sigmoid(np.array([[-1.0, -1.0]])),
                ecm_layer_min_sigm(input_logits))
            # sample size 2
            input_logits = tf.constant(np.array([[-1.0, 2.0],
                                                 [2.0, -1.0]]),
                                       dtype=tf.float32)
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
            ecm_layer_min = layers.ExtremumConstraintModule(
                activation="linear",
                extremum="min",
                adjacency_matrix=adj_mat,
                sparse_adjacency=is_sparse_mat)
            # transpose adj_mat for max to give a parent->child adj_mat
            ecm_layer_max = layers.ExtremumConstraintModule(
                activation="linear",
                extremum="max",
                adjacency_matrix=adj_mat.transpose(),
                sparse_adjacency=is_sparse_mat)
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
            if is_sparse_mat:
                self.assertShapeEqual(np.zeros(shape=[1, 4, 4]),
                                      tf.sparse.to_dense(
                                          ecm_layer_max.adjacency_mat))
            else:
                self.assertShapeEqual(np.zeros(shape=[1, 4, 4]),
                                      ecm_layer_max.adjacency_mat)

            # Try feeding an input layer (functional API)
            ecm_layer_max = layers.ExtremumConstraintModule(
                activation="linear",
                extremum="max",
                adjacency_matrix=adj_mat.transpose(),
                sparse_adjacency=is_sparse_mat)
            input = keras.layers.Input(shape=(4,))
            dense = keras.layers.Dense(4,
                                       #kernel_initializer='ones',
                                       #bias_initializer='zeros'
                                       )(input)
            ecm_layer_max(dense)
            # Compile and fit in a non learning model
            ecm_layer_max = layers.ExtremumConstraintModule(
                activation="linear",
                extremum="max",
                adjacency_matrix=adj_mat.transpose(),
                sparse_adjacency=is_sparse_mat)
            model = keras.Sequential([keras.layers.Input(shape=(4,)),
                                      ecm_layer_max])
            model.compile(loss=keras.losses.binary_crossentropy)
            y = np.array([[4.0, 3.0, -1.0, 2.0],
                          [4.0, 3.0, -1.0, 3.0],
                          [3.0, 3.0, -1.0, 3.0]])
            model.fit(x=input_logits, y=y, verbose=False)

            # Compile and fit in a learning model
            # only if not sparse since sparce.reduce_max has no gradient
            # implemented
            if not is_sparse_mat:
                ecm_layer_max = layers.ExtremumConstraintModule(
                    activation="linear",
                    extremum="max",
                    adjacency_matrix=adj_mat.transpose(),
                    sparse_adjacency=is_sparse_mat)
                model = keras.Sequential([keras.layers.Input(shape=(4,)),
                                          keras.layers.Dense(4,
                                                             #kernel_initializer='ones',
                                                             #bias_initializer='zeros'
                                                             ),
                                          ecm_layer_max])
                model.compile(loss=keras.losses.binary_crossentropy)
                y = np.array([[4.0, 3.0, -1.0, 2.0],
                              [4.0, 3.0, -1.0, 3.0],
                              [3.0, 3.0, -1.0, 3.0]])
                model.fit(x=input_logits, y=y, verbose=False)
