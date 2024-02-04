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


class TestCompoundMetric(TestCase):
    def test_inmodel(self):
        x_train = [[0], [1], [2], [3]]
        y_train = [0, 0, 1, 1]
        model = keras.Sequential(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
                      metrics=("accuracy", "AUC"))

        compound_met = callbacks.CompoundMetric({"accuracy": 1,
                                                 "auc": 2})

        hist: keras.callbacks.History = model.fit(x=x_train,
                                                  y=y_train,
                                                  epochs=1,
                                                  callbacks=[compound_met],
                                                  verbose=False)
        self.assertAlmostEqual(
            (1*hist.history['accuracy'][0] + 2*hist.history['auc'][0])/3.0,
            hist.history['accuracy1_auc2'][0]
        )

        compound_met = callbacks.CompoundMetric({"accuracy": 1,
                                                 "auc": 2},
                                                compound_metric_name="blob")
        hist: keras.callbacks.History = model.fit(x=x_train,
                                                  y=y_train,
                                                  epochs=1,
                                                  callbacks=[compound_met],
                                                  verbose=False)
        self.assertAlmostEqual(
            (1*hist.history['accuracy'][0] + 2*hist.history['auc'][0])/3.0,
            hist.history['blob'][0]
        )

        # provide an empty prefix
        compound_met = callbacks.CompoundMetric({"accuracy": 1,
                                                 "auc": 2},
                                                met_prefixes='',
                                                compound_metric_name="blob")
        hist: keras.callbacks.History = model.fit(x=x_train,
                                                  y=y_train,
                                                  epochs=1,
                                                  callbacks=[compound_met],
                                                  verbose=False)
        self.assertAlmostEqual(
            (1*hist.history['accuracy'][0] + 2*hist.history['auc'][0])/3.0,
            hist.history['blob'][0]
        )

        # provide a list of prefixes
        compound_met = callbacks.CompoundMetric({"accuracy": 1,
                                                 "auc": 2},
                                                met_prefixes=['', 'val_'],
                                                compound_metric_name="blob")
        x_test = [[3], [2], [1], [0]]
        y_test = [0, 0, 1, 1]
        hist: keras.callbacks.History = model.fit(x=x_train,
                                                  y=y_train,
                                                  validation_data=(x_test,y_test),
                                                  epochs=1,
                                                  callbacks=[compound_met],
                                                  verbose=False)
        self.assertAlmostEqual(
            (1*hist.history['accuracy'][0] + 2*hist.history['auc'][0])/3.0,
            hist.history['blob'][0]
        )
        self.assertAlmostEqual(
            (1*hist.history['val_accuracy'][0] + 2*hist.history['val_auc'][0])/3.0,
            hist.history['val_blob'][0]
        )
