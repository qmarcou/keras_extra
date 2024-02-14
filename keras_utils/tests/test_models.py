from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_utils import models


class TestSequentialPreOutputLoss(tf.test.TestCase):

    def test___init___(self):
        SequentialPreOutputLoss = models.SequentialPreOutputLoss
        inputL = keras.layers.InputLayer(input_shape=1, name="input")
        inputT = keras.Input(shape=1)
        outputL = keras.layers.Dense(1, name="output")
        hiddenL = keras.layers.Dense(1, name="hidden")
        SequentialPreOutputLoss()
        SequentialPreOutputLoss(layers=inputL)
        SequentialPreOutputLoss(layers=outputL)
        SequentialPreOutputLoss(layers=[inputL, outputL])
        m = SequentialPreOutputLoss(layers=[inputL, hiddenL, outputL])
        self.assertEqual(1, len(m.outputs))
        # check that a second output is not created when the specified loss
        # layer is the output layer
        m = SequentialPreOutputLoss(layers=[inputL, hiddenL, outputL],
                                    loss_layer_name=outputL.name)
        self.assertEqual(1, len(m.outputs))
        m = SequentialPreOutputLoss(layers=[inputL, hiddenL, outputL],
                                    loss_layer_name="hidden")
        self.assertEqual(2, len(m.outputs))
        # this should work upon model construction but fail at compile
        # since the loss layer could be added later via self.add before compile
        m = SequentialPreOutputLoss(layers=[inputL, hiddenL, outputL],
                                    loss_layer_name="ZZZ")
        self.assertEqual(1, len(m.outputs))
        # Test model instantiation with Input tensor instead of InputLayer
        # SequentialPreOutputLoss(layers=inputT)
        SequentialPreOutputLoss(layers=[inputT, outputL])
        m = SequentialPreOutputLoss(layers=[inputT, outputL],
                                    loss_layer_name="output")
        m = SequentialPreOutputLoss(layers=[inputT, hiddenL, outputL],
                                    loss_layer_name="output")
        m = SequentialPreOutputLoss(layers=[inputT, hiddenL, outputL],
                                    loss_layer_name="hidden")
        # Test model instantiation without input layer
        m = SequentialPreOutputLoss(layers=[outputL])
        m = SequentialPreOutputLoss(layers=[outputL],
                                    loss_layer_name="output")
        self.assertRaises(ValueError, m.build)
        m.build(input_shape=(None, 1))
        self.assertEqual(1, len(m.outputs))
        m = SequentialPreOutputLoss(
            layers=[hiddenL, outputL],
            loss_layer_name="hidden")
        m.build(input_shape=(None, 1))
        self.assertEqual(2, len(m.outputs))
        m = SequentialPreOutputLoss(layers=[hiddenL, outputL],
                                    loss_layer_name="output")
        m.build(input_shape=(None, 1))
        self.assertEqual(1, len(m.outputs))

    def test_compile(self):
        SequentialPreOutputLoss = models.SequentialPreOutputLoss
        inputL = keras.layers.InputLayer(input_shape=1, name="input")
        outputL = keras.layers.Dense(1, name="output")
        hiddenL = keras.layers.Dense(1, name="hidden")
        m = SequentialPreOutputLoss(layers=[inputL, hiddenL, outputL],
                                    loss_layer_name="hidden")
        m.compile()
        m.compile(loss=keras.losses.BinaryCrossentropy(),
                  metrics=keras.metrics.AUC())
        # Non existing loss layer name
        m = SequentialPreOutputLoss(layers=[inputL, hiddenL, outputL],
                                    loss_layer_name="ZZZ")
        self.assertRaises(ValueError, m.compile)

    def test_evaluate(self):
        SequentialPreOutputLoss = models.SequentialPreOutputLoss
        inputL = keras.layers.InputLayer(input_shape=1, name="input")
        intermL = keras.layers.Dense(1, name="linear", activation="linear")
        outputL = keras.layers.Activation(name="output",
                                          activation="sigmoid")
        m = SequentialPreOutputLoss(layers=[inputL, intermL, outputL],
                                    loss_layer_name="linear")
        m.build()
        intermL.set_weights(weights=[np.array([[1]]),
                                     np.array([0])])
        m.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False))
        ev = m.evaluate(x=np.array([[0], [1]]),
                        y=np.array([[0], [1]]),
                        verbose=False,
                        return_dict=True)
        # This is a bit of a hack test
        self.assertEqual(0.0, ev['loss'])
        m.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=keras.metrics.MeanAbsoluteError())
        ev2 = m.evaluate(x=np.array([[0], [1]]),
                         y=np.array([[0], [1]]),
                         verbose=False,
                         return_dict=True)
        self.assertNotEqual(0.0, ev2['loss'])
        # m.fit(x=np.array([[0], [1]]),
        #       y=np.array([[0], [1]]), epochs=3)

        m2 = keras.Sequential(layers=[inputL, intermL, outputL])
        m2.build()
        intermL.set_weights(weights=[np.array([[1]]),
                                     np.array([0])])
        m2.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
                   metrics=keras.metrics.MeanAbsoluteError())
        ev = m2.evaluate(x=np.array([[0], [1]]),
                         y=np.array([[0], [1]]),
                         verbose=False,
                         return_dict=True)
        # Assert that the loss is the same if computed with logits option
        # on previous layer than a Sequential model with loss computed on non
        # logits
        self.assertEqual(ev2['loss'], ev['loss'])
        # Assert that the computed metrics are the same
        self.assertEqual(ev2['output_mean_absolute_error'],
                         ev['mean_absolute_error'])

