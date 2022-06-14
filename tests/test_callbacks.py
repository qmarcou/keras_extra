from unittest import TestCase
from keras_utils import callbacks
from tensorflow import keras
import keras


class TestDataEvaluator(TestCase):
    def test_inmodel(self):
        x_train = [[0], [1], [2], [3]]
        y_train = [0, 0, 1, 1]
        model = keras.Sequential(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", metrics="accuracy")
        x_test = [[3], [2], [1], [0]]
        y_test = [0, 0, 1, 1]
        data_eval = callbacks.DataEvaluator(x=x_test, y=y_test,
                                            eval_prefix="dataEvalTester_")
        hist: keras.callbacks.History = model.fit(x=x_train,
                                                  y=y_train,
                                                  epochs=3,
                                                  callbacks=[data_eval],
                                                  verbose=False)
        eval = model.evaluate(x=x_test,
                              y=y_test, return_dict=True)
        self.assertAlmostEqual(hist.history['dataEvalTester_loss'][-1],
                               eval['loss'])
