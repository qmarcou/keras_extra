import keras
import keras_utils.models
from keras_utils import losses
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import Reduction
from numpy.testing import assert_array_almost_equal
import numpy as np
import tensorflow as tf


class TestWBCE(tf.test.TestCase):
    def test_weighted_binary_crossentropy(self):
        wbc = losses.weighted_binary_crossentropy
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]  # shape nsamples*noutput
        weights = [[1.0, 1.0], [1.0, 1.0]]  # shape noutput*2
        assert_array_almost_equal(binary_crossentropy(y_true, y_pred,
                                                      from_logits=False),
                                  wbc(y_true, y_pred, weights,
                                      from_logits=False))
        # Check that the same weights can be broadcast
        weights = 1.0
        assert_array_almost_equal(binary_crossentropy(y_true, y_pred,
                                                      from_logits=False),
                                  wbc(y_true, y_pred, weights,
                                      from_logits=False))
        weights = [1.0, 1.0]
        assert_array_almost_equal(binary_crossentropy(y_true, y_pred,
                                                      from_logits=False),
                                  wbc(y_true, y_pred, weights,
                                      from_logits=False))
        # Check that class_weights with last dimension > 3 raises an exception
        weights = [1.0, 1.0, 1.0]
        self.assertRaises(tf.errors.InvalidArgumentError,
                          wbc, y_true, y_pred, weights,
                          from_logits=False)
        weights = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        self.assertRaises(tf.errors.InvalidArgumentError,
                          wbc, y_true, y_pred, weights,
                          from_logits=False)
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

    def test_WeightedBinaryCrossentropy(self):
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]  # shape nsamples*noutput
        weights = [[1.0, 1.0], [1.0, 3.0]]
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
        # Inside a model pipeline check that the loss function handles
        # correctly tensors with unknown dimension (at model initialization
        # batch size is unknown and the shape of the first dimension is None
        # and the passed vectors are placeholders
        wbce = losses.WeightedBinaryCrossentropy(class_weights=weights,
                                                 from_logits=False,
                                                 reduction=Reduction.SUM_OVER_BATCH_SIZE)
        model = keras.Sequential([keras.layers.Dense(units=2,
                                                     activation='sigmoid')])
        model.compile(loss=wbce)
        model.fit(x=[[1], [2]],
                  y=y_true,
                  verbose=False)
        # Inside a more functional model
        model = keras_utils.models.SequentialPreOutputLoss([
            keras.layers.InputLayer(input_shape=(1,)),
            keras.layers.Dense(units=2, activation='linear', name='dense'),
            keras.layers.Activation(activation="sigmoid", name='output')
        ], loss_layer_name='dense')
        wbce = losses.WeightedBinaryCrossentropy(class_weights=weights,
                                                 from_logits=True,
                                                 reduction=Reduction.SUM_OVER_BATCH_SIZE)
        model.compile(loss=wbce)
        model.fit(x=[[1], [2]],
                  y=y_true,
                  verbose=False)


class TestMCLoss(tf.test.TestCase):
    def test_call(self):
        adj_mat = np.array([[0, 0], [1, 0]])  # 1 is parent of 0
        weights = np.array([[1.0, 1.0],
                            [1.0, 1.0]])
        mcl = losses.MCLoss(adjacency_matrix=adj_mat,
                            from_logits=True,
                            activation='linear',
                            class_weights=None)
        wbc = losses.WeightedBinaryCrossentropy(from_logits=True,
                                                class_weights=weights)

        y_true = tf.constant(np.array([[0.0, 1.0]]), dtype=tf.float32)

        input_logits = tf.constant(np.array([[-1.0, 2.0]]), dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            mcl(y_pred=input_logits, y_true=y_true)
        )
        input_logits = tf.constant(np.array([[2.0, -1.0]]), dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            mcl(y_pred=input_logits, y_true=y_true)
        )
        y_true = tf.constant(np.array([[1.0, 1.0]]), dtype=tf.float32)
        input_logits = tf.constant(np.array([[-1.0, 2.0]]), dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            mcl(y_pred=input_logits, y_true=y_true)
        )
        input_logits = tf.constant(np.array([[2.0, -1.0]]), dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=np.array([[2.0, 2.0]]), y_true=y_true),
            mcl(y_pred=input_logits, y_true=y_true)
        )
        y_true = tf.constant(np.array([[0.0, 0.0]]), dtype=tf.float32)
        input_logits = tf.constant(np.array([[-1.0, 2.0]]), dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            mcl(y_pred=input_logits, y_true=y_true)
        )
        input_logits = tf.constant(np.array([[2.0, -1.0]]), dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=np.array([[2.0, 2.0]]), y_true=y_true),
            mcl(y_pred=input_logits, y_true=y_true)
        )
        # Test casting of inputs if they have mismatched dtypes
        input_logits = tf.constant(input_logits, dtype=tf.float32)
        y_true = tf.constant(np.array([[0.0, 0.0]]), dtype=tf.int8)
        mcl(y_pred=input_logits, y_true=y_true)

        # Test casting to the cross ent type when both input have another dtype
        # TODO
        # input_logits = tf.constant(np.array([[2.0, 1.0]]), dtype=tf.int8)
        # y_true = tf.constant(np.array([[0.0, 0.0]]), dtype=tf.int8)
        # mcl(y_pred=input_logits, y_true=y_true)

    def test_inmodel(self):
        adj_mat = np.array([[0, 0], [1, 0]])  # 1 is parent of 0
        weights = np.array([[1.0, 1.0],
                            [1.0, 1.0]])
        mcl = losses.MCLoss(adjacency_matrix=adj_mat,
                            from_logits=True,
                            activation='linear',
                            class_weights=weights)
        model = keras.models.Sequential(layers=[
            keras.layers.Dense(units=2, activation="linear")
        ])
        model.compile(loss=mcl)
        y_true = tf.constant(np.array([[1.0, 1.0]]), dtype=tf.float32)
        x = np.random.random_sample(size=(1, 2))
        model.fit(x=x,
                  y=y_true,
                  verbose=False)


class TestTreeMinLoss(tf.test.TestCase):
    def test_call(self):
        # We test the following tree:
        #       4
        #      / \
        #     2   3
        #    / \
        #   0   1
        # Here I only test y_true combinations respecting the hierarchy, note
        # that no check is made regarding this at runtime
        adj_mat = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],  # 0 and 1 are leaves
                            [1, 1, 0, 0, 0],  # 2 is parent of 0 and 1
                            [0, 0, 0, 0, 0],  # 3 is a leaf
                            [1, 1, 1, 1, 0]],  # 4 is parent of 2 and 3
                           )  # 3 is an upper leaf
        weights = np.array([1.0])  # equal weights for all cases
        tml = losses.TreeMinLoss(adjacency_matrix=adj_mat,
                                 from_logits=True,
                                 activation='linear',
                                 class_weights=None)
        wbc = losses.WeightedBinaryCrossentropy(from_logits=True,
                                                class_weights=weights)

        # Leaf 1 is true
        y_true = tf.constant(np.array([[0.0, 1.0, 1.0, 0.0, 1.0]]),
                             dtype=tf.float32)
        ## Coherent prediction
        input_logits = tf.constant(np.array([[-1.0, 2.0, 2.5, -0.5, 3.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        ## Incoherent predictions for 2
        input_logits = tf.constant(np.array([[-1.0, 2.0, 1.5, -0.5, 3.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=np.array([[-1.0, 1.5, 1.5, -0.5, 3.0]]), y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        ## Incoherent predictions for 4
        input_logits = tf.constant(np.array([[-1.0, 2.0, 2.5, -0.5, 1.5]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=np.array([[-1.0, 1.5, 1.5, -0.5, 1.5]]), y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        input_logits = tf.constant(np.array([[-1.0, 2.0, 2.5, -0.5, -1.5]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=np.array([[-1.0, -1.5, -1.5, -0.5, -1.5]]), y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )

        # Leaf 3 is true
        y_true = tf.constant(np.array([[0.0, 0.0, 0.0, 1.0, 1.0]]),
                             dtype=tf.float32)
        ## Coherent prediction
        input_logits = tf.constant(np.array([[-1.0, -2.0, 0, 2.5, 3.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        ## Incoherent prediction for 2 and 4
        input_logits = tf.constant(np.array([[-1.0, 0.0, -1.0, 2.5, 2.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=np.array([[-1.0, 0.0, 0.0, 2.0, 2.0]]), y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        # Node 2 is true
        y_true = tf.constant(np.array([[0.0, 0.0, 1.0, 0.0, 1.0]]),
                             dtype=tf.float32)
        ## Coherent prediction
        input_logits = tf.constant(np.array([[-1.0, -2.0, 0, -2.5, 3.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        input_logits = tf.constant(np.array([[-1.0, 1.0, 2.0, -2.5, 3.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        ## Incoherent prediction for 2 and 4
        input_logits = tf.constant(np.array([[-1.0, 4.0, 3.0, -2.5, 2.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=np.array([[-1.0, 4.0, 2.0, -2.5, 2.0]]), y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        # Node 4 is true
        y_true = tf.constant(np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]),
                             dtype=tf.float32)
        ## Coherent prediction
        input_logits = tf.constant(np.array([[-1.0, -2.0, 0, -2.5, 3.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        ## Incoherent prediction for 1,0 and 3
        input_logits = tf.constant(np.array([[-1.0, .0, -2.0, 3.5, 3.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=np.array([[-1.0, .0, 0.0, 3.5, 3.0]]), y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        # No true node
        y_true = tf.constant(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
                             dtype=tf.float32)
        ## Coherent prediction
        input_logits = tf.constant(np.array([[-1.0, -2.0, 0, -2.5, 3.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=input_logits, y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        ## Incoherent prediction for 2 and 1
        input_logits = tf.constant(np.array([[-1.0, 3.0, 2.0, -2.5, 0.0]]),
                                   dtype=tf.float32)
        self.assertAllCloseAccordingToType(
            wbc(y_pred=np.array([[-1.0, 3.0, 3.0, -2.5, 3.0]]), y_true=y_true),
            tml(y_pred=input_logits, y_true=y_true)
        )
        # y_true = tf.constant(np.array([[1.0, 0.0]]) would violate the
        # hierarchy
        # TODO raise an error?

        # Test casting of inputs if they have mismatched dtypes
        input_logits = tf.constant(np.array([[-1.0, 3.0, 2.0, -2.5, 0.0]]),
                                   dtype=tf.float32)
        y_true = tf.constant(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
                             dtype=tf.int8)
        tml(y_pred=input_logits, y_true=y_true)

        # Test casting to the cross ent type when both input have another dtype
        # TODO
        # input_logits = tf.constant(np.array([[2.0, 1.0]]), dtype=tf.int8)
        # y_true = tf.constant(np.array([[0.0, 0.0]]), dtype=tf.int8)
        # mcl(y_pred=input_logits, y_true=y_true)

    def test_inmodel(self):
        adj_mat = np.array([[0, 0], [1, 0]])  # 1 is parent of 0
        weights = np.array([[1.0, 1.0],
                            [1.0, 1.0]])
        tml = losses.TreeMinLoss(adjacency_matrix=adj_mat,
                                 from_logits=True,
                                 activation='linear',
                                 class_weights=weights)
        model = keras.models.Sequential(layers=[
            keras.layers.Dense(units=2, activation="linear")
        ])
        model.compile(loss=tml)
        y_true = tf.constant(np.array([[1.0, 1.0]]), dtype=tf.float32)
        x = np.random.random_sample(size=(1, 2))
        model.fit(x=x,
                  y=y_true,
                  verbose=False)
