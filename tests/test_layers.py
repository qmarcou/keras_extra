from unittest import TestCase
import tensorflow as tf
from keras_utils import layers
import numpy as np
from tensorflow import keras
from keras_utils.layers import DenseHierL2Reg, \
    _dense_compute_hier_weight_diff_tensor
from scipy import sparse


class TestECM(tf.test.TestCase):
    def test_call(self):
        adj_mat = np.array([[0, 0], [1, 0]])  # 1 is child of 0
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

        # Test different input types
        input_logits = tf.cast(input_logits, dtype=tf.float64)
        ecm_layer_max(input_logits)
        # FIXME support for int in sparse ECM
        # input_logits = tf.cast(input_logits, dtype=tf.int64)
        # ecm_layer_max(input_logits)
        # input_logits = tf.cast(input_logits, dtype=tf.int32)
        # ecm_layer_max(input_logits)

    def test_build(self):
        # 4 class adjacency matrix, test coherent classification cases
        # 1 and 2 are child of 0
        # 3 is child of 1 (hence of 0)
        adj_mat = np.array([[0, 0, 0, 0], [1, 0, 0, 0],
                            [1, 0, 0, 0], [1, 1, 0, 0]])
        input_logits = tf.constant(np.array([[4.0, 3.0, -1.0, 2.0],
                                             [4.0, 2.0, -1.0, 3.0],
                                             [-1.0, 2.0, -1.0, 3.0]]),
                                   dtype=tf.float32)
        for is_sparse_mat in [False, True]:
            ecm_layer_min = layers.ExtremumConstraintModule(
                activation="linear",
                extremum="min",
                adjacency_matrix=adj_mat
                .transpose(),
                sparse_adjacency=is_sparse_mat)
            # Test building with partially unknown input_shape
            ecm_layer_min.build(input_shape=tf.TensorShape([None, 4]))
            # Test building with partially unknown input_shape
            ecm_layer_min.build(input_shape=tf.TensorShape([None, 4]))

            # Try feeding an input layer (functional API)
            ecm_layer_max = layers.ExtremumConstraintModule(
                activation="linear",
                extremum="max",
                adjacency_matrix=adj_mat.transpose(),
                sparse_adjacency=is_sparse_mat)
            input = keras.layers.Input(shape=(4,))
            dense = keras.layers.Dense(4,
                                       # kernel_initializer='ones',
                                       # bias_initializer='zeros'
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

            ecm_layer_max = layers.ExtremumConstraintModule(
                activation="linear",
                extremum="max",
                adjacency_matrix=adj_mat.transpose(),
                sparse_adjacency=is_sparse_mat)
            model = keras.Sequential([keras.layers.Input(shape=(4,)),
                                      keras.layers.Dense(4,
                                                         # kernel_initializer='ones',
                                                         # bias_initializer='zeros'
                                                         ),
                                      ecm_layer_max])
            model.compile(loss=keras.losses.binary_crossentropy)
            y = np.array([[4.0, 3.0, -1.0, 2.0],
                          [4.0, 3.0, -1.0, 3.0],
                          [3.0, 3.0, -1.0, 3.0]])
            model.evaluate(x=input_logits, y=y, verbose=False)
            if not is_sparse_mat:
                model.fit(x=input_logits, y=y, verbose=False)

    def test_is_input_hier(self):
        # 4 class adjacency matrix, test coherent classification cases
        # 1 and 2 are child of 0
        # 3 is child of 1 (hence of 0)
        adj_mat = np.array([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [1, 0, 0, 0],
                            [1, 1, 0, 0]])
        ecm_layer_min = layers.ExtremumConstraintModule(activation="linear",
                                                        extremum="min",
                                                        adjacency_matrix=adj_mat,
                                                        sparse_adjacency=False)
        self.assertTrue(ecm_layer_min.is_input_hier_coherent(np.array(
            [[0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 1, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 0, 1],
             [1, 1, 1, 1]], dtype=np.float64),
            collapse_all=True))
        self.assertAllEqual([False, False, False, False, False, False],
                            ecm_layer_min.is_input_hier_coherent(np.array(
                                [[0, 0, 0, 1],
                                 [1, 0, 0, 1],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 1, 1, 1]], dtype=np.float64),
                                collapse_all=False))

        # Test with non binary values
        self.assertAllEqual([False, True],
                            ecm_layer_min.is_input_hier_coherent(np.array(
                                [[0.2, 0, 0, 1.5],
                                 [1.5, 1.4, 1.5, 1.0]], dtype=np.float64),
                                collapse_all=False))


class TestDenseHierL2Reg(tf.test.TestCase):
    def test_init(self):
        adj_mat = np.array([[0, 0], [1, 0]])  # 1 is child of 0
        DenseHierL2Reg(units=1,
                       adjacency_matrix=adj_mat,
                       hier_side="out",
                       regularization_factor=.1,
                       tree_like=True
                       )
        self.assertRaises(ValueError,
                          DenseHierL2Reg,
                          units=1,
                          adjacency_matrix=adj_mat,
                          hier_side="out",
                          regularization_factor=-.1,
                          tree_like=True)
        adj_mat = np.array([[0, 1], [1, 0]])  # 1 is child of 0
        # FIXME adj mat consistentcy test failing
        self.assertRaises(ValueError,
                          DenseHierL2Reg,
                          units=1,
                          adjacency_matrix=adj_mat,
                          hier_side="out",
                          regularization_factor=.1,
                          tree_like=True)
        DenseHierL2Reg(units=1,
                       adjacency_matrix=adj_mat,
                       hier_side="out",
                       regularization_factor=.1,
                       tree_like=False
                       )
        adj_mat = np.array([[0, 0.2], [1, 0]])  # 1 is child of 0
        self.assertRaises(ValueError,
                          DenseHierL2Reg,
                          units=1,
                          adjacency_matrix=adj_mat,
                          hier_side="out",
                          regularization_factor=.1,
                          tree_like=False)

    def test_build(self):
        adj_mat = np.array([[0, 0], [1, 0]])  # 1 is child of 0
        hier_dense = DenseHierL2Reg(units=1,
                                    adjacency_matrix=adj_mat,
                                    hier_side="out",
                                    regularization_factor=.1,
                                    tree_like=True
                                    )
        self.assertRaises(ValueError,
                          hier_dense.build,
                          input_shape=tf.TensorShape([None, 2])
                          )
        hier_dense = DenseHierL2Reg(units=2,
                                    adjacency_matrix=adj_mat,
                                    hier_side="out",
                                    regularization_factor=.1,
                                    tree_like=True
                                    )
        hier_dense.build(tf.TensorShape([None, 1]))
        hier_dense = DenseHierL2Reg(units=3,
                                    adjacency_matrix=adj_mat,
                                    hier_side="out",
                                    regularization_factor=.1,
                                    tree_like=True
                                    )
        self.assertRaises(ValueError,
                          hier_dense.build,
                          input_shape=tf.TensorShape([None, 2])
                          )
        # test input side hierarchy
        hier_dense = DenseHierL2Reg(units=3,
                                    adjacency_matrix=adj_mat,
                                    hier_side="in",
                                    regularization_factor=.1,
                                    tree_like=True
                                    )
        self.assertRaises(ValueError,
                          hier_dense.build,
                          input_shape=tf.TensorShape([None, 1])
                          )
        self.assertRaises(ValueError,
                          hier_dense.build,
                          input_shape=tf.TensorShape([None, 3])
                          )
        hier_dense.build(tf.TensorShape([None, 2]))

    def test_call(self):
        adj_mat = np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 0, 0]])  # 1 is child of 0
        input_tensor = tf.constant([[.0, .0, .0],
                                    [.0, .0, .0]])

        # Test output regularization
        hier_dense = DenseHierL2Reg(units=3,
                                    adjacency_matrix=adj_mat,
                                    hier_side="out",
                                    regularization_factor=1.0,
                                    tree_like=True
                                    )
        hier_dense.build(input_shape=tf.TensorShape([None, 3]))

        # set kernel and bias to controlled values
        # the expected loss is the squared l2 norm of the difference between
        # unit 0 and unit 1 vectors of parameters including bias
        hier_dense.set_weights([np.array([[1.0, 2.0, 20.0],
                                          [3.0, 5.0, 20.0],
                                          [4.0, 6.0, 20.0]]),
                                np.array([1.0, 2.0, 20.0])])
        hier_dense(input_tensor)  # call
        self.assertAllCloseAccordingToType(
            np.array([10.0]),
            hier_dense.losses)

        # Test input regularization
        hier_dense = DenseHierL2Reg(units=3,
                                    adjacency_matrix=adj_mat,
                                    hier_side="in",
                                    regularization_factor=1.0,
                                    tree_like=True
                                    )
        hier_dense.build(input_shape=tf.TensorShape([None, 3]))
        # set weights to controlled values
        # bias is an output activation property and should not be accounted for
        # in input regularization
        hier_dense.set_weights([np.array([[1.0, 2.0, 3.0],
                                          [3.0, 5.0, 4.0],
                                          [4.0, 6.0, 5.0]]),
                                np.array([20.0, 20.0, 20.0])])
        hier_dense(input_tensor)  # call
        self.assertAllCloseAccordingToType(
            np.array([14.0]),
            hier_dense.losses)
        hier_dense._clear_losses()

        # Check that the direction of the adjacency matrix does not matter
        hier_dense2 = DenseHierL2Reg(units=3,
                                     adjacency_matrix=adj_mat.transpose(),
                                     hier_side="in",
                                     regularization_factor=1.0,
                                     tree_like=True)
        hier_dense2.build(input_shape=tf.TensorShape([None, 3]))
        hier_dense.set_weights(hier_dense2.get_weights())  # random weights
        hier_dense(input_tensor)  # call
        hier_dense2(input_tensor)  # call
        self.assertAllCloseAccordingToType(
            hier_dense2.losses,
            hier_dense.losses)

        self.assertAllCloseAccordingToType(
            hier_dense2.losses,
            hier_dense.losses)

    def test_inmodel(self):
        adj_mat = np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 0, 0]])  # 1 is child of 0
        hier_dense = DenseHierL2Reg(units=3,
                                    adjacency_matrix=adj_mat,
                                    hier_side="out",
                                    regularization_factor=1.0,
                                    tree_like=True
                                    )
        model = keras.Sequential([keras.layers.Input(shape=(4,)),
                                  keras.layers.Dense(3),
                                  hier_dense])
        model.compile(loss=keras.losses.binary_crossentropy)

        x = tf.constant(np.array([[4.0, 3.0, -1.0, 2.0],
                                  [4.0, 2.0, -1.0, 3.0],
                                  [-1.0, 2.0, -1.0, 3.0]]),
                        dtype=tf.float32)

        y = np.array([[4.0, 3.0, -1.0],
                      [4.0, 3.0, -1.0],
                      [3.0, 3.0, -1.0]])
        hist: keras.callbacks.History = model.fit(x=x, y=y, verbose=False)

        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        hier_dense = DenseHierL2Reg(units=3,
                                    adjacency_matrix=adj_mat,
                                    hier_side="out",
                                    regularization_factor=1.0,
                                    tree_like=True
                                    )
        model = keras.Sequential([keras.layers.Input(shape=(4,)),
                                  keras.layers.Dense(3),
                                  hier_dense])
        model.compile(loss=keras.losses.binary_crossentropy)
        hist: keras.callbacks.History = model.fit(x=x, y=y, verbose=False)


class Test(tf.test.TestCase):
    def test_dense_compute_hier_weight_diff_vector(self):
        weights = tf.constant([[1.0, 2.0, 4.0],
                               [3.0, 5.0, 6.0]],
                              dtype=tf.float32)
        adj_list = tf.constant([[0, 1],
                                [1, 0]], dtype=tf.int32)

        self.assertAllCloseAccordingToType(
            np.array([[-2.0, -3.0, -2.0],
                      [2.0, 3.0, 2.0]]),
            _dense_compute_hier_weight_diff_tensor(weights=weights,
                                                   adj_list=adj_list,
                                                   axis=0))
        self.assertAllCloseAccordingToType(
            np.array([[-2.0, -3.0, -2.0],
                      [2.0, 3.0, 2.0]]),
            _dense_compute_hier_weight_diff_tensor(weights=weights,
                                                   adj_list=adj_list,
                                                   axis=-2))
        self.assertAllCloseAccordingToType(
            np.array([[-1.0, -2.0],
                      [1.0, 2.0]]),
            _dense_compute_hier_weight_diff_tensor(weights=weights,
                                                   adj_list=adj_list,
                                                   axis=1))
        self.assertAllCloseAccordingToType(
            np.array([[-1.0, -2.0],
                      [1.0, 2.0]]),
            _dense_compute_hier_weight_diff_tensor(weights=weights,
                                                   adj_list=adj_list,
                                                   axis=-1))

    def test__ragged_coo_graph_reduce(self):
        # Check 2D input (case of most interest)
        x = [[.2, .3, .5],
             [.5, .2, .3]]
        adj_list = [[0, 0],
                    [1, 1],
                    [2, 2],
                    [0, 1]]
        self.assertAllCloseAccordingToType(
            [[0.2, 0.3, 0.5],
             [0.5, 0.5, 0.3]],
            layers._ragged_coo_graph_reduce(values=x,
                                            adjacency_list=adj_list,
                                            axis=1,
                                            reduce_fn=tf.reduce_max,
                                            sorted_adj_list=False)
        )
        self.assertAllCloseAccordingToType(
            [[0.2, 0.2, 0.5],
             [0.5, 0.2, 0.3]],
            layers._ragged_coo_graph_reduce(values=x,
                                            adjacency_list=adj_list,
                                            axis=1,
                                            reduce_fn=tf.reduce_min,
                                            sorted_adj_list=False)
        )
        self.assertAllCloseAccordingToType(
            [[0.2, 0.5, 0.5],
             [0.5, 0.7, 0.3]],
            layers._ragged_coo_graph_reduce(values=x,
                                            adjacency_list=adj_list,
                                            axis=1,
                                            reduce_fn=tf.reduce_sum,
                                            sorted_adj_list=False)
        )
        # Check 1D
        self.assertAllCloseAccordingToType(
            [0.2, 0.5, 0.5],
            layers._ragged_coo_graph_reduce(values=x[0],
                                            adjacency_list=adj_list,
                                            axis=0,
                                            reduce_fn=tf.reduce_sum,
                                            sorted_adj_list=False)
        )
        # Check 3D
        x = [[[.2, .3, .5],
              [.5, .2, .3]],
             [[.2, .3, .5],
              [.5, .2, .3]]]

        self.assertAllCloseAccordingToType(
            [[[0.2, 0.5, 0.5],
             [0.5, 0.7, 0.3]],
             [[0.2, 0.5, 0.5],
              [0.5, 0.7, 0.3]]],
            layers._ragged_coo_graph_reduce(values=x,
                                            adjacency_list=adj_list,
                                            axis=2,
                                            reduce_fn=tf.reduce_sum,
                                            sorted_adj_list=False)
        )
        adj_list = [[0, 0],
                    [1, 1],
                    [0, 1]]
        self.assertAllCloseAccordingToType(
            [[[.2, .3, .5],
              [.7, .5, .8]],
             [[.2, .3, .5],
              [.7, .5, .8]]],
            layers._ragged_coo_graph_reduce(values=x,
                                            adjacency_list=adj_list,
                                            axis=1,
                                            reduce_fn=tf.reduce_sum,
                                            sorted_adj_list=False)
        )