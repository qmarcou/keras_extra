"""A collection of custom metrics functions and classes."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Type


def get_lowest_true_index(y_true, y_pred, from_logits=True):
    """
    Get the lowest rank of a true label inside the prediction vector.

    This function returns how far, for each sample,a learning
    algorithm needs to go down in the ordered list of predicted labels to cover
    all the true labels of an instance.

    Parameters
    ----------
    y_true An array or tensor of shape (sample_size,...,Number of labels)
    y_pred An array or tensor of same shape than y_true
    from_logits bool If true will take into account potential negative values
     in y_pred. If not set to true and negative values are passed behavior
     is undefined.

    Returns
    -------
    A tensor of shape (sample_size,...), the -1 dimension of y_true/y_pred
    (number of labels) is aggregated.
    """
    # assert np.all(tf.logical_or(tf.equal(y_true, 1), tf.equal(y_true, 0)))
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    y_pred.shape.assert_is_compatible_with(y_true.shape)

    # get the minimal y_pred value for true y_true
    hack = tf.divide(y_pred, y_true)  # cast non true pred values to infinity
    # in case logits where passed we need to set -inf to inf
    if from_logits:
        comp = tf.equal(hack, -np.inf)
        hack = (tf.where(comp,
                         tf.multiply(tf.ones_like(hack), np.inf),
                         hack))
    min_true_pred = tf.reduce_min(hack,
                                  axis=-1, keepdims=True)

    lowest_true_rank = tf.reduce_sum(
        tf.cast(tf.greater_equal(y_pred, min_true_pred), tf.uint64),
        axis=-1)
    return lowest_true_rank


def stateless_coverage(y_true, y_pred, sample_weight=None,
                       from_logits: bool = True):
    """
    Compute the coverage for the sample

    Coverage is the metric that evaluates how far on average, a learning
    algorithm needs to go down in the ordered list of predicted labels to cover
    all the true labels of an instance. Clearly, the smaller the value of
    coverage, the better the performance.

    Parameters
    ----------
    y_true An array or tensor of shape (sample_size,...,Number of labels)
    y_pred An array or tensor of same shape than y_true
    sample_weight None or a vector/tensor of weights
    from_logits bool If true will take into account potential negative values
     in y_pred. If not set to true and negative values are passed behavior
     is undefined.

    Returns
    -------
    A scalar
    """
    lowest_true_rank = get_lowest_true_index(y_true, y_pred,
                                             from_logits=from_logits)
    if sample_weight is not None:
        lowest_true_rank = tf.multiply(lowest_true_rank, sample_weight)
    return (tf.reduce_mean(tf.cast(lowest_true_rank,
                                   tf.float32)))


class Coverage(keras.metrics.Mean):
    """
    A Metric subclass to compute coverage.

    This class wraps call to the stateless_coverage function to provide a
    Keras Metric subclass.
    """

    # TODO: check -1 from Tarekegn et al
    # TODO: check how to compute coverage for different levels in hierarchy
    # TODO: make sure I expect the correct dims from y_true/y_pred
    # TODO: normalize by number of true labels?
    def __init__(self, name='coverage', dtype=None, from_logits=True):
        self.from_logits = from_logits
        super(Coverage, self).__init__(name=name,
                                       dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        lowest_true_rank = get_lowest_true_index(y_true, y_pred,
                                                 from_logits=self.from_logits)
        super(Coverage, self).update_state(lowest_true_rank,
                                           sample_weight=sample_weight)


def subset_metric_builder(metric_class: type[keras.metrics.Metric]):
    # TODO:
    #  check if I can assign dynamic name to the created class, cf
    #  https://python-course.eu/oop/dynamically-creating-classes-with-type.php

    # Dynamic class building using
    # https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
    class SubsetMetricWrapper(metric_class):
        def __init__(self, slicing_func_kwargs, name=None, slicing_func=tf.gather,
                     **kwargs):
            self.gather_kwargs = slicing_func_kwargs
            self.slicing_func = slicing_func
            if name is None:
                name = ("subset_" + self.__class__.__base__.__name__)
            super(SubsetMetricWrapper, self).__init__(name=name,
                                                      **kwargs)

        def update_state(self, y_true, y_pred, sample_weight=None):
            # Filter data to the defined subset
            y_true = self.slicing_func(params=y_true, **self.gather_kwargs)
            y_pred = self.slicing_func(params=y_pred, **self.gather_kwargs)
            return (super(SubsetMetricWrapper, self).
                    update_state(y_true=y_true, y_pred=y_pred,
                                 sample_weight=sample_weight))

    return SubsetMetricWrapper
