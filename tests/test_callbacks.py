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
        data_eval = callbacks.DataEvaluator(x=x_train, y=y_train,
                                            eval_prefix="dataEvalTester_")
        x_test = [[3], [2], [1], [0]]
        y_test = [0, 0, 1, 1]
        hist: keras.callbacks.History = model.fit(x=x_test,
                                                  y=y_test,
                                                  epochs=3,
                                                  callbacks=[data_eval],)

        print(hist.history)
