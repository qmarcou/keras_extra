from unittest import TestCase
from numpy import testing
import keras.metrics
import tensorflow as tf
from keras_utils import metrics
import numpy as np


class TestCoverage(TestCase):
    def testcoverage(self):
        y_true = [[0, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]]
        y_pred = [[.2, .3, .4, .5], [0.2, 0.5, .4, .6], [.8, .3, .9, .8]]
        # Expected ranks: 3, 3, 3
        m = metrics.Coverage()
        m.update_state(y_true, y_pred)
        self.assertEqual(float(m.result()), 3.0)
        # make sure aggregation is correctly performed
        m.update_state(y_true, y_pred)
        self.assertEqual(3.0, float(m.result()))

        # Test behavior when there is no true label
        # Maybe I should expect nan instead
        y_true = [[0, 0, 0, 0]]
        y_pred = [[.2, .3, .4, .5]]
        m = metrics.Coverage()
        m.update_state(y_true, y_pred)
        self.assertEqual(0.0, float(m.result()))

        # Test behavior when there are some negative pred values
        # Should be used with from_logits = True only
        y_true = [[0, 1, 0, 1]]
        y_pred = [[-10, -2, .4, .5]]
        m = metrics.Coverage(from_logits=True)
        m.update_state(y_true, y_pred)
        self.assertEqual(3.0, float(m.result()))


class TestRankAtPercentile(tf.test.TestCase):
    def test_rank_at_percentile(self):
        y_true = [[0, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]
        y_pred = [[-3, -2, -1, 0, 1, 2, 3],
                  [3, 2, 1, 0, -1, -2, -3],
                  [3, 2, 1, 0, -1, -2, -3]]

        self.assertAllEqual(np.array([4, 3, 1]),
                            metrics.rank_at_percentile(y_true, y_pred, q=50,
                                                       no_true_label_value=1.0,
                                                       interpolation='linear'))

    def test_metric(self):
        y_true = [[0, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]
        y_pred = [[-3, -2, -1, 0, 1, 2, 3],
                  [3, 2, 1, 0, -1, -2, -3],
                  [3, 2, 1, 0, -1, -2, -3]]
        rap = metrics.RankAtPercentile(q=50, no_true_label_value=1.0,
                                       interpolation='linear')

        rap.update_state(y_true, y_pred)
        self.assertAllCloseAccordingToType(2.666666, rap.result())


class RankErrorsAtPercentile(tf.test.TestCase):
    def test_rank_errors_at_percentile(self):
        y_true = tf.constant([[0, 1, 0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0]])
        y_pred = tf.constant([[-3, -2, -1, 0, 1, 2, 3],
                              [3, 2, 1, 0, -1, -2, -3],
                              [3, 2, 1, 0, -1, -2, -3]])

        self.assertAllEqual(np.array([2, 1, 0]),
                            metrics.rank_errors_at_percentile(
                                y_true, y_pred, q=50,
                                no_true_label_value=1.0,
                                interpolation='linear'))

    def test_class(self):
        y_true = [[0, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]
        y_pred = [[-3, -2, -1, 0, 1, 2, 3],
                  [3, 2, 1, 0, -1, -2, -3],
                  [3, 2, 1, 0, -1, -2, -3]]
        reap = metrics.RankErrorsAtPercentile(q=50, no_true_label_value=1.0,
                                              interpolation='linear')

        reap.update_state(y_true, y_pred)
        self.assertEqual(1.0, reap.result())


class TestSubsetMetric(TestCase):
    def test_metric_construction(self):
        new_metric = metrics.subset_metric_builder(keras.metrics.Accuracy)
        self.assertIsInstance(new_metric, type)
        new_metric_instance = new_metric(slicing_func_kwargs={})
        self.assertEqual(new_metric_instance.name, "subset_Accuracy")
        new_metric_instance = new_metric(slicing_func_kwargs={},
                                         name="mysubsetmetric")
        self.assertEqual(new_metric_instance.name,
                         "mysubsetmetric")

    def test_metric_subsetting(self):
        new_metric = metrics.subset_metric_builder(
            keras.metrics.BinaryAccuracy)
        normal_metric = keras.metrics.BinaryAccuracy()
        new_metric_instance = new_metric(slicing_func=tf.gather,
                                         slicing_func_kwargs={
                                             'indices': [0, 1], 'axis': 0})
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0, 1], [1, 0]]
        testing.assert_array_almost_equal(
            normal_metric(y_true=y_true, y_pred=y_pred),
            new_metric_instance(y_true=y_true, y_pred=y_pred))

        # Slice out batch examples
        new_metric_instance = new_metric(slicing_func=tf.gather,
                                         slicing_func_kwargs={
                                             'indices': [1], 'axis': 0})
        normal_metric.reset_states()
        self.assertEqual(normal_metric(y_true=[0, 0], y_pred=[1, 0]),
                         new_metric_instance(y_true=y_true, y_pred=y_pred))

        # Slice out output dimensions
        new_metric_instance = new_metric(slicing_func=tf.gather,
                                         slicing_func_kwargs={
                                             'indices': [0], 'axis': 1})
        normal_metric.reset_states()
        self.assertEqual(normal_metric(y_true=[0, 0], y_pred=[0, 1]),
                         new_metric_instance(y_true=y_true, y_pred=y_pred))
