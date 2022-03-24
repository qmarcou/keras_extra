from unittest import TestCase
import tensorflow as tf
from tensorflow import keras
from keras_utils import models


class TestSequentialPreOutputLoss(tf.test.TestCase):

    def test___init___(self):
        SequentialPreOutputLoss = models.SequentialPreOutputLoss
        inputL = keras.layers.InputLayer(input_shape=1, name="input")
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


    def test_compile(self):
        SequentialPreOutputLoss = models.SequentialPreOutputLoss
        inputL = keras.layers.InputLayer(input_shape=1, name="input")
        output = keras.layers.Dense(1, name="output")
        hidden = keras.layers.Dense(1, name="hidden")
        a = SequentialPreOutputLoss(layers=[inputL, hidden, output],
                                    loss_layer_name="hidden")
        a.compile()
        a.compile(loss=keras.losses.BinaryCrossentropy(),
                  metrics=keras.metrics.AUC())
