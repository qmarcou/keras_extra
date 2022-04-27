"""A collection of utility functions."""
import tensorflow as tf

# Adapted from tensorflow-ranking v0.6.0, utils.py
# Last commit to file a2f2d1433523ed2bf0bb3308b533d48e275d7fd0
# Copyright 2022 The TensorFlow Ranking Authors.
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

_PADDING_LABEL = -1.
_PADDING_PREDICTION = -1e6
_PADDING_WEIGHT = 0.


def is_label_valid(labels):
    """Returns a boolean `Tensor` for label validity."""
    labels = tf.convert_to_tensor(value=labels)
    return tf.greater_equal(labels, 0.)


def _get_shuffle_indices(shape, mask=None, shuffle_ties=True, seed=None):
    """Gets indices which would shuffle a tensor.
  Args:
    shape: The shape of the indices to generate.
    mask: An optional mask that indicates which entries to place first. Its
      shape should be equal to given shape.
    shuffle_ties: Whether to randomly shuffle ties.
    seed: The ops-level random seed.
  Returns:
    An int32 `Tensor` with given `shape`. Its entries are indices that would
    (randomly) shuffle the values of a `Tensor` of given `shape` along the last
    axis while placing masked items first.
  """
    # Generate random values when shuffling ties or all zeros when not.
    if shuffle_ties:
        shuffle_values = tf.random.uniform(shape, seed=seed)
    else:
        shuffle_values = tf.zeros(shape, dtype=tf.float32)

    # Since shuffle_values is always in [0, 1), we can safely increase entries
    # where mask=False with 2.0 to make sure those are placed last during the
    # argsort op.
    if mask is not None:
        shuffle_values = tf.where(mask, shuffle_values, shuffle_values + 2.0)

    # Generate indices by sorting the shuffle values.
    return tf.argsort(shuffle_values, stable=True)


def sort_by_scores(scores,
                   features_list,
                   topn=None,
                   shuffle_ties=True,
                   seed=None,
                   mask=None):
    """Sorts list of features according to per-example scores.
  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    features_list: A list of `Tensor`s to be sorted. The shape of the `Tensor`
      can be [batch_size, list_size] or [batch_size, list_size, feature_dims].
      The latter is applicable for example features.
    topn: An integer as the cutoff of examples in the sorted list.
    shuffle_ties: A boolean. If True, randomly shuffle before the sorting.
    seed: The ops-level random seed used when `shuffle_ties` is True.
    mask: An optional `Tensor` of shape [batch_size, list_size] representing
      which entries are valid for sorting. Invalid entries will be pushed to the
      end.
  Returns:
    A list of `Tensor`s as the list of sorted features by `scores`.
  """
    with tf.compat.v1.name_scope(name='sort_by_scores'):
        scores = tf.cast(scores, tf.float32)
        scores.get_shape().assert_has_rank(2)
        list_size = tf.shape(input=scores)[1]
        if topn is None:
            topn = list_size
        topn = tf.minimum(topn, list_size)

        # Set invalid entries (those whose mask value is False) to the minimal value
        # of scores so they will be placed last during sort ops.
        if mask is not None:
            scores = tf.where(mask, scores, tf.reduce_min(scores))

        # Shuffle scores to break ties and/or push invalid entries (according to
        # mask) to the end.
        shuffle_ind = None
        if shuffle_ties or mask is not None:
            shuffle_ind = _get_shuffle_indices(
                tf.shape(input=scores), mask, shuffle_ties=shuffle_ties,
                seed=seed)
            scores = tf.gather(scores, shuffle_ind, batch_dims=1, axis=1)

        # Perform sort and return sorted feature_list entries.
        _, indices = tf.math.top_k(scores, topn, sorted=True)
        if shuffle_ind is not None:
            indices = tf.gather(shuffle_ind, indices, batch_dims=1, axis=1)
        return [tf.gather(f, indices, batch_dims=1, axis=1) for f in
                features_list]


def ragged_to_dense(labels, predictions, weights):
    """Converts given inputs from ragged tensors to dense tensors.
  Args:
    labels: A `tf.RaggedTensor` of the same shape as `predictions` representing
      relevance.
    predictions: A `tf.RaggedTensor` with shape [batch_size, (list_size)]. Each
      value is the ranking score of the corresponding example.
    weights: An optional `tf.RaggedTensor` of the same shape of predictions or a
      `tf.Tensor` of shape [batch_size, 1]. The former case is per-example and
      the latter case is per-list.
  Returns:
    A tuple (labels, predictions, weights, mask) of dense `tf.Tensor`s.
  """
    # TODO: Add checks to validate (ragged) shapes of input tensors.
    mask = tf.cast(tf.ones_like(labels).to_tensor(0.), dtype=tf.bool)
    labels = labels.to_tensor(_PADDING_LABEL)
    if predictions is not None:
        predictions = predictions.to_tensor(_PADDING_PREDICTION)
    if isinstance(weights, tf.RaggedTensor):
        weights = weights.to_tensor(_PADDING_WEIGHT)
    return labels, predictions, weights, mask
