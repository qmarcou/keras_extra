"""A collection of custom metrics functions and classes."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional
from keras_utils import utils, stats
import abc
import six

K = tf.keras.backend


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


def rank_at_percentile(y_true: tf.Tensor, y_pred: tf.Tensor, q,
                       no_true_label_value=None,
                       interpolation='linear') -> tf.Tensor:
    y_true = tf.cast(y_true, dtype=tf.bool)

    pred_ranks = utils.compute_ranks(values=y_pred, axis=-1,
                                     direction='DESCENDING',
                            stable=False)
    pred_ranks = tf.cast(pred_ranks, dtype=tf.float32)

    template = tf.fill(dims=tf.shape(y_true), value=np.nan)
    # Set NaN rank for y_true=0
    masked_ranks = tf.where(condition=y_true, x=pred_ranks, y=template)

    # Get rank at the specified percentile
    ranks = stats.nanpercentile(
        x=masked_ranks,
        q=q,
        axis=-1,
        interpolation=interpolation)

    if no_true_label_value is not None:
        # Filter possible NaNs and set the value to the requested filling value
        mask = tf.math.is_nan(ranks)
        filler = tf.fill(value=no_true_label_value,
                         dims=tf.shape(ranks))
        ranks = tf.where(condition=mask, x=filler,
                         y=ranks)
    return ranks


def rank_errors_at_percentile(y_true, y_pred, q,
                              no_true_label_value=None,
                              interpolation='linear') -> tf.Tensor:
    # First compute the rank of the true label at the given percentile
    ranks = rank_at_percentile(y_true, y_pred,
                               q=q,
                               no_true_label_value=no_true_label_value,
                               interpolation=interpolation)

    # Now subtract the number of true labels at the corresponding
    # percentile to get the number of errors at this percentile
    true_labs_at_p = rank_at_percentile(y_true, y_true,
                                        q=q,
                                        no_true_label_value=no_true_label_value,
                                        interpolation=interpolation)
    return ranks - true_labs_at_p


class Coverage(keras.metrics.Mean):
    """
    A Metric subclass to compute coverage.

    This class wraps call to the stateless_coverage function to provide a
    Keras Metric subclass.
    """

    # TODO: check -1 from Tarekegn et al
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


class RankAtPercentile(keras.metrics.Mean):
    """
    A Metric subclass to compute the average rank at a given percentile.

    This class wraps call to the rank_at_percentile function to provide a
    Keras Metric subclass. It returns the average rank at a specified
    percentile of true labels. E.g when 4 labels are true and q=75 (third
    quartile), it will return the rank of the 3 label.
    """

    def __init__(self, q, no_true_label_value=1.0, interpolation='linear',
                 name='rankatpercentile', dtype=None, from_logits=True):
        self.from_logits = from_logits
        # FIXME check that q and notruevalue are scalars
        self.percentile = q
        self.fill_value = no_true_label_value
        self.interpolation = interpolation
        super(RankErrorsAtPercentile, self).__init__(name=name,
                                                     dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        ranks = rank_at_percentile(y_true, y_pred,
                                   q=self.percentile,
                                   no_true_label_value=self.fill_value,
                                   interpolation=self.interpolation)

        super(RankErrorsAtPercentile, self).update_state(ranks,
                                                         sample_weight=sample_weight)


class RankErrorsAtPercentile(keras.metrics.Mean):
    """
    A Metric subclass to compute the average rank at a given percentile.

    This class wraps call to the rank_at_percentile function to provide a
    Keras Metric subclass. It returns the average rank at a specified
    percentile of true labels. E.g when 4 labels are true and q=75 (third
    quartile), it will return the rank of the 3 label.
    """

    def __init__(self, q, no_true_label_value=1.0, interpolation='linear',
                 name='rankatpercentile', dtype=None, from_logits=True):
        self.from_logits = from_logits
        # FIXME check that q and notruevalue are scalars
        self.percentile = q
        self.fill_value = no_true_label_value
        self.interpolation = interpolation
        super(RankErrorsAtPercentile, self).__init__(name=name,
                                                     dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        rank_errors = rank_errors_at_percentile(
            y_true=y_true, y_pred=y_pred,
            q=self.percentile,
            no_true_label_value=self.fill_value,
            interpolation=self.interpolation)

        super(RankErrorsAtPercentile, self).update_state(rank_errors,
                                                         sample_weight=sample_weight)


def subset_metric_builder(metric_class: type[keras.metrics.Metric]):
    # TODO:
    #  check if I can assign dynamic name to the created class, cf
    #  https://python-course.eu/oop/dynamically-creating-classes-with-type.php

    # Dynamic class building using
    # https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
    class SubsetMetricWrapper(metric_class):
        def __init__(self, slicing_func_kwargs, name=None,
                     slicing_func=tf.gather,
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


# Adapted From tensorflow-addons v0.16., metrics.f_scores.py
# last commit to file f30df4322b5580b3e5946530a60f7126035dd73b
# Type hints/checks were removed to get rid of overall package dependency
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements F scores."""


@tf.keras.utils.register_keras_serializable(package="Addons")
class FBetaScore(tf.keras.metrics.Metric):
    r"""Computes F-Beta score.
    It is the weighted harmonic mean of precision
    and recall. Output range is `[0, 1]`. Works for
    both multi-class and multi-label classification.
    $$
    F_{\beta} = (1 + \beta^2) * \frac{\textrm{precision} * \textrm{recall}}{(\beta^2 \cdot \textrm{precision}) + \textrm{recall}}
    $$
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.
    Returns:
        F-Beta Score: float.
    Raises:
        ValueError: If the `average` has values other than
        `[None, 'micro', 'macro', 'weighted']`.
        ValueError: If the `beta` value is less than or equal
        to 0.
    `average` parameter behavior:
        None: Scores for each class are returned.
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    Usage:
    >>> metric = tfa.metrics.FBetaScore(num_classes=3, beta=2.0, threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    array([0.3846154 , 0.90909094, 0.8333334 ], dtype=float32)
    """

    def __init__(
            self,
            num_classes,
            average: Optional[str] = None,
            beta=1.0,
            threshold=None,
            name: str = "fbeta_score",
            dtype=None,
            **kwargs,
    ):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Unknown average type. Acceptable values "
                "are: [None, 'micro', 'macro', 'weighted']"
            )

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float")

        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError(
                    "The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros",
                dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    def _weighted_sum(self, val, sample_weight):
        if sample_weight is not None:
            val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
        return tf.reduce_sum(val, axis=self.axis)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold,
                                    tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        self.true_positives.assign_add(
            self._weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(
            self._weighted_sum(y_pred * (1 - y_true), sample_weight)
        )
        self.false_negatives.assign_add(
            self._weighted_sum((1 - y_pred) * y_true, sample_weight)
        )
        self.weights_intermediate.assign_add(
            self._weighted_sum(y_true, sample_weight))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate,
                tf.reduce_sum(self.weights_intermediate)
            )
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        reset_value = tf.zeros(self.init_shape, dtype=self.dtype)
        K.batch_set_value([(v, reset_value) for v in self.variables])

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()


@tf.keras.utils.register_keras_serializable(package="Addons")
class F1Score(FBetaScore):
    r"""Computes F-1 Score.
    It is the harmonic mean of precision and recall.
    Output range is `[0, 1]`. Works for both multi-class
    and multi-label classification.
    $$
    F_1 = 2 \cdot \frac{\textrm{precision} \cdot \textrm{recall}}
    {\textrm{precision} + \textrm{recall}}
    $$
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro`
            and `weighted`. Default value is None.
        threshold: Elements of `y_pred` above threshold are
            considered to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.
    Returns:
        F-1 Score: float.
    Raises:
        ValueError: If the `average` has values other than
        [None, 'micro', 'macro', 'weighted'].
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    Usage:
    >>> metric = keras_utils.metrics.F1Score(num_classes=3, threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    array([0.5      , 0.8      , 0.6666667], dtype=float32)
    """

    def __init__(
            self,
            num_classes,
            average: str = None,
            threshold=None,
            name: str = "f1_score",
            dtype=None,
    ):
        super().__init__(num_classes, average, 1.0, threshold, name=name,
                         dtype=dtype)

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


##############################################################################
#                         END OF tensorflow-addons code                      #
##############################################################################


class MicroF1score(F1Score):
    def __init__(
            self,
            num_classes,
            threshold=None,
            name: str = "micro_f1_score",
            dtype=None,
    ):
        super().__init__(num_classes=num_classes,
                         average="micro",
                         threshold=threshold,
                         name=name,
                         dtype=dtype)


class MacroF1score(F1Score):
    def __init__(
            self,
            num_classes,
            threshold=None,
            name: str = "macro_f1_score",
            dtype=None,
    ):
        super().__init__(num_classes=num_classes,
                         average="macro",
                         threshold=threshold,
                         name=name,
                         dtype=dtype)


# Adapted From tensorflow-ranking v0.6.0, metrics_impl.py
# last commit to file d8bd34c4d0006e0d142fe854a40b0e3a193f85da
# Copyright 2021 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

_DEFAULT_GAIN_FN = lambda label: tf.pow(2.0, label) - 1
_DEFAULT_RANK_DISCOUNT_FN = lambda rank: tf.math.log(2.) / tf.math.log1p(rank)


class _RankingMetric(six.with_metaclass(abc.ABCMeta, object)):
    """Interface for ranking metrics."""

    def __init__(self, ragged=False):
        """Constructor.
    Args:
      ragged: A bool indicating whether the supplied tensors are ragged. If
        True labels, predictions and weights (if providing per-example weights)
        need to be ragged tensors with compatible shapes.
    """
        self._ragged = ragged

    @property
    @abc.abstractmethod
    def name(self):
        """The metric name."""
        raise NotImplementedError('Calling an abstract method.')

    def _prepare_and_validate_params(self, labels, predictions, weights, mask):
        """Prepares and validates the parameters.
    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means
        a relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.
      mask: A `Tensor` of the same shape as predictions indicating which entries
        are valid for computing the metric.
    Returns:
      (labels, predictions, weights, mask) ready to be used for metric
      calculation.
    """
        if any(isinstance(tensor, tf.RaggedTensor)
               for tensor in [labels, predictions, weights]):
            raise ValueError('labels, predictions and/or weights are ragged '
                             'tensors, use ragged=True to enable ragged '
                             'support for metrics.')
        labels = tf.convert_to_tensor(value=labels)
        predictions = tf.convert_to_tensor(value=predictions)
        weights = 1.0 if weights is None else tf.convert_to_tensor(
            value=weights)
        example_weights = tf.ones_like(labels) * weights
        predictions.get_shape().assert_is_compatible_with(
            example_weights.get_shape())
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        predictions.get_shape().assert_has_rank(2)

        # All labels should be >= 0. Invalid entries are reset.
        if mask is None:
            mask = utils.is_label_valid(labels)
        labels = tf.compat.v1.where(mask, labels, tf.zeros_like(labels))
        predictions = tf.compat.v1.where(
            mask, predictions, -1e-6 * tf.ones_like(predictions) +
                               tf.reduce_min(input_tensor=predictions, axis=1,
                                             keepdims=True))
        return labels, predictions, example_weights, mask

    def compute(self, labels, predictions, weights=None, mask=None):
        """Computes the metric with the given inputs.
    Args:
      labels: A `Tensor` of the same shape as `predictions` representing
        relevance.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: An optional `Tensor` of the same shape of predictions or
        [batch_size, 1]. The former case is per-example and the latter case is
        per-list.
      mask: An optional `Tensor` of the same shape as predictions indicating
        which entries are valid for computing the metric. Will be ignored if
        the metric was constructed with ragged=True.
    Returns:
      A tf metric.
    """
        if self._ragged:
            labels, predictions, weights, mask = utils.ragged_to_dense(
                labels, predictions, weights)
        labels, predictions, weights, mask = self._prepare_and_validate_params(
            labels, predictions, weights, mask)
        return self._compute_impl(labels, predictions, weights, mask)

    @abc.abstractmethod
    def _compute_impl(self, labels, predictions, weights, mask):
        """Computes the metric with the given inputs.
    Args:
      labels: A `Tensor` of the same shape as `predictions` representing
        relevance.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1].
        The former case is per-example and the latter case is per-list.
      mask: A `Tensor` of the same shape as predictions indicating which entries
        are valid for computing the metric.
    Returns:
      A tf metric.
    """
        raise NotImplementedError('Calling an abstract method.')


class NDCGMetric(_RankingMetric):
    """Implements normalized discounted cumulative gain (NDCG)."""

    def __init__(self,
                 name,
                 topn,
                 gain_fn=_DEFAULT_GAIN_FN,
                 rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN,
                 ragged=False):
        """Constructor."""
        super(NDCGMetric, self).__init__(ragged=ragged)
        self._name = name
        self._topn = topn
        self._gain_fn = gain_fn
        self._rank_discount_fn = rank_discount_fn

    @property
    def name(self):
        """The metric name."""
        return self._name

    def _compute_impl(self, labels, predictions, weights, mask):
        """See `_RankingMetric`."""
        topn = tf.shape(predictions)[1] if self._topn is None else self._topn
        sorted_labels, sorted_weights = utils.sort_by_scores(
            predictions, [labels, weights], topn=topn, mask=mask)
        dcg = _discounted_cumulative_gain(sorted_labels, sorted_weights,
                                          self._gain_fn,
                                          self._rank_discount_fn)
        # Sorting over the weighted labels to get ideal ranking.
        ideal_sorted_labels, ideal_sorted_weights = utils.sort_by_scores(
            weights * labels, [labels, weights], topn=topn, mask=mask)
        ideal_dcg = _discounted_cumulative_gain(ideal_sorted_labels,
                                                ideal_sorted_weights,
                                                self._gain_fn,
                                                self._rank_discount_fn)
        per_list_ndcg = tf.compat.v1.math.divide_no_nan(dcg, ideal_dcg)
        per_list_weights = _per_example_weights_to_per_list_weights(
            weights=weights,
            relevance=self._gain_fn(tf.cast(labels, dtype=tf.float32)))
        return per_list_ndcg,


def _per_example_weights_to_per_list_weights(weights, relevance):
    """Computes per list weight from per example weight.
  The per-list weights are computed as:
    per_list_weights = sum(weights * relevance) / sum(relevance).
  For a list with sum(relevance) = 0, we set a default weight as the following
  average weight while all the lists with sum(weights) = 0 are ignored.
    sum(per_list_weights) / num(sum(relevance) != 0 && sum(weights) != 0)
  When all the lists have sum(relevance) == 0, we set the average weight to 1.0.
  Such a computation is good for the following scenarios:
    - When all the weights are 1.0, the per list weights will be 1.0 everywhere,
      even for lists without any relevant examples because
        sum(per_list_weights) ==  num(sum(relevance) != 0)
      This handles the standard ranking metrics where the weights are all 1.0.
    - When every list has a nonzero weight, the default weight is not used. This
      handles the unbiased metrics well.
    - For the mixture of the above 2 scenario, the weights for lists with
      nonzero relevance and nonzero weights is proportional to
        per_list_weights / sum(per_list_weights) *
        num(sum(relevance) != 0) / num(lists).
      The rest have weights 1.0 / num(lists).
  Args:
    weights:  The weights `Tensor` of shape [batch_size, list_size].
    relevance:  The relevance `Tensor` of shape [batch_size, list_size].
  Returns:
    The per list `Tensor` of shape [batch_size, 1]
  """
    nonzero_weights = tf.greater(
        tf.reduce_sum(input_tensor=weights, axis=1, keepdims=True), 0.0)
    per_list_relevance = tf.reduce_sum(
        input_tensor=relevance, axis=1, keepdims=True)
    nonzero_relevance = tf.compat.v1.where(
        nonzero_weights,
        tf.cast(tf.greater(per_list_relevance, 0.0), tf.float32),
        tf.zeros_like(per_list_relevance))
    nonzero_relevance_count = tf.reduce_sum(
        input_tensor=nonzero_relevance, axis=0, keepdims=True)

    per_list_weights = tf.compat.v1.math.divide_no_nan(
        tf.reduce_sum(input_tensor=weights * relevance, axis=1, keepdims=True),
        per_list_relevance)
    sum_weights = tf.reduce_sum(
        input_tensor=per_list_weights, axis=0, keepdims=True)

    avg_weight = tf.compat.v1.where(
        tf.greater(nonzero_relevance_count, 0.0),
        tf.compat.v1.math.divide_no_nan(sum_weights, nonzero_relevance_count),
        tf.ones_like(nonzero_relevance_count))
    return tf.compat.v1.where(
        nonzero_weights,
        tf.where(
            tf.greater(per_list_relevance, 0.0), per_list_weights,
            tf.ones_like(per_list_weights) * avg_weight),
        tf.zeros_like(per_list_weights))


def _discounted_cumulative_gain(labels,
                                weights=None,
                                gain_fn=_DEFAULT_GAIN_FN,
                                rank_discount_fn=_DEFAULT_RANK_DISCOUNT_FN):
    """Computes discounted cumulative gain (DCG).
  DCG = SUM(gain_fn(label) / rank_discount_fn(rank)). Using the default values
  of the gain and discount functions, we get the following commonly used
  formula for DCG: SUM((2^label -1) / log(1+rank)).
  Args:
    labels: The relevance `Tensor` of shape [batch_size, list_size]. For the
      ideal ranking, the examples are sorted by relevance in reverse order. In
      alpha_dcg, it is a `Tensor` with shape [batch_size, list_size,
      subtopic_size].
    weights: A `Tensor` of the same shape as labels or [batch_size, 1]. The
      former case is per-example and the latter case is per-list.
    gain_fn: (function) Transforms labels.
    rank_discount_fn: (function) The rank discount function.
  Returns:
    A `Tensor` as the weighted discounted cumulative gain per-list. The
    tensor shape is [batch_size, 1].
  """
    list_size = tf.shape(input=labels)[1]
    position = tf.cast(tf.range(1, list_size + 1), dtype=tf.float32)
    gain = gain_fn(tf.cast(labels, dtype=tf.float32))
    discount = rank_discount_fn(position)
    return tf.reduce_sum(
        input_tensor=weights * gain * discount, axis=1, keepdims=True)
##############################################################################
#                         END OF tensorflow-ranking code                     #
##############################################################################
