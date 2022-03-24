from unittest import TestCase
import tensorflow as tf
from tensorflow import keras
from keras_utils import models


class TestSequentialPreOutputLoss(tf.test.TestCase):

    def test___init___(self):
        SequentialPreOutputLoss = models.SequentialPreOutputLoss
        inputL = keras.layers.InputLayer(input_shape=1, name="input")
        output = keras.layers.Dense(1, name="output")
        hidden = keras.layers.Dense(1, name="hidden")
        SequentialPreOutputLoss()
        SequentialPreOutputLoss(layers=inputL)
        SequentialPreOutputLoss(layers=output)
        SequentialPreOutputLoss(layers=[inputL, output])
        SequentialPreOutputLoss(layers=[inputL, hidden, output])
        SequentialPreOutputLoss(layers=[inputL, hidden, output],
                                loss_layer_name="output")
        SequentialPreOutputLoss(layers=[inputL, hidden, output],
                                loss_layer_name="hidden")

    def test__add_loss_output(self):
        self.fail()

    def test_add(self):
        self.fail()

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
        print(a.outputs)
        print(a.output)