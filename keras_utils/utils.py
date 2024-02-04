"""A collection of utility functions."""
import tensorflow as tf
from enum import Enum


def not_in(x, list):
    """Check if values in x are contained in list"""
    raise NotImplementedError


def is_in(x, list):
    raise NotImplementedError


class _SideDim(Enum):
    First = "first",
    Last = "last"


def _move_axis_to_side_dim(x, axis, side: _SideDim) -> tf.Tensor:
    """
    Move the given axes to first or last dimensions
    Parameters
    ----------
    x the tf.Tensor of interest
    axis 0 or 1D tf.Tensor (or similar) specifying the axes that should be
        moved to first or last last dimension. If several axes are given they will be
        ordered according to their order in the 'axis' argument. Axis values
        must be in the range -rank:rank.
    side: "last" or "first"
    Returns
    -------
    A tf.Tensor with same rank as x but reordered dimensions.
    """
    axes_range = tf.range(tf.rank(x), dtype=tf.int32)

    # Prepare axis to 1D tensor
    # Will throw an error if axis is not a scalar or 1D tensor
    axis = tf.convert_to_tensor(axis, dtype=tf.int32)
    axis = tf.reshape(axis, shape=(-1,))

    # Get actual axis indices if any axis is <0
    neg_axis = tf.less(axis, 0)
    if tf.reduce_any(neg_axis):
        axis = tf.where(condition=neg_axis, x=tf.rank(x) + axis, y=axis)

    # Create new axis index map and transpose the tensor accordingly
    mask = tf.reduce_all(tf.not_equal(
        tf.reshape(axes_range, shape=(-1, 1)),
        tf.reshape(axis, shape=(1, -1))),
        axis=1)  # boils down to not_in in 1D

    masked_range = tf.boolean_mask(
        tensor=axes_range,
        mask=mask)

    if side == _SideDim.Last:
        axis_map = tf.concat([masked_range,
                              axis], axis=0)
    elif side == _SideDim.First:
        axis_map = tf.concat([axis,
                              masked_range], axis=0)
    else:
        raise ValueError("side can be either first or last")

    return tf.transpose(a=x, perm=axis_map)


def move_axis_to_last_dim(x, axis) -> tf.Tensor:
    """
    Move the given axes to last dimensions
    Parameters
    ----------
    x the tf.Tensor of interest
    axis 0 or 1D tf.Tensor (or similar) specifying the axes that should be
        moved to last dimension. If several axes are given they will be
        ordered according to their order in the 'axis' argument.
    Returns
    -------
    A tf.Tensor with same rank as x but reordered dimensions.
    """
    return _move_axis_to_side_dim(x=x, axis=axis, side=_SideDim.Last)


def move_axis_to_first_dim(x, axis) -> tf.Tensor:
    """
    Move the given axes to first dimensions
    Parameters
    ----------
    x the tf.Tensor of interest
    axis 0 or 1D tf.Tensor (or similar) specifying the axes that should be
        moved to first dimension. If several axes are given they will be
        ordered according to their order in the 'axis' argument.
    Returns
    -------
    A tf.Tensor with same rank as x but reordered dimensions.
    """
    return _move_axis_to_side_dim(x=x, axis=axis, side=_SideDim.First)


def move_axis_to(input_tensor, axis_index, new_index):
    """
    Move a tensor axis to the specified index.

    Only works with 1 axis at a time.

    Parameters
    ----------
    input_tensor
    axis_index: index of the axis to me moved
    new_index: destination index

    Returns
    -------
    tf.Tensor with modified axes
    """
    # TODO make it work with more than 1 axis
    axes_range = tf.range(tf.rank(input_tensor), dtype=tf.int32)

    # Prepare axis to 1D tensor
    # Will throw an error if axis is not a scalar or 1D tensor
    axis = tf.convert_to_tensor(axis_index, dtype=tf.int32)
    axis = tf.squeeze(axis)

    dest_index = tf.convert_to_tensor(new_index, dtype=tf.int32)
    dest_index = tf.squeeze(new_index)

    tf.assert_rank(axis, 0, message="Moving more than one axis at a time is "
                                    "not supported.")
    tf.assert_rank(dest_index, 0,
                   message="Moving more than one axis at a time is "
                           "not supported.")

    # Get actual axis indices if any axis is <0
    # TODO cleanup code redundancy via _to_positive_indices function
    neg_axis = tf.less(axis, 0)
    if tf.reduce_any(neg_axis):
        axis = tf.where(condition=neg_axis,
                        x=tf.rank(input_tensor) + axis,
                        y=axis)

    neg_axis = tf.less(dest_index, 0)
    if tf.reduce_any(neg_axis):
        dest_index = tf.where(condition=neg_axis,
                              x=tf.rank(input_tensor) + dest_index,
                              y=dest_index)

    # Create new axis index map and transpose the tensor accordingly
    mask = tf.reduce_all(tf.not_equal(
        tf.reshape(axes_range, shape=(-1, 1)),
        tf.reshape(axis, shape=(1, -1))),
        axis=1)  # boils down to not_in in 1D

    masked_range = tf.boolean_mask(
        tensor=axes_range,
        mask=mask)

    axis_map = tf.concat([masked_range[0:dest_index],
                          tf.reshape(axis, shape=(1,)),
                          masked_range[dest_index:]],
                         axis=0)

    return tf.transpose(a=input_tensor, perm=axis_map)


def _to_positive_indices(tensor_shape, indices):
    # FIXME only works in 1D
    # tf.reshape(tensor_shape)
    # Get actual axis indices if any axis is <0
    neg_axis = tf.less(indices, 0)
    if tf.reduce_any(neg_axis):
        pos_indices = tf.where(condition=neg_axis,
                               x=tensor_shape + indices,
                               y=indices)
    return pos_indices


def swap_axes(input_tensor, axis_1, axis_2):
    raise NotImplementedError


def shape_without_axis(input_tensor: tf.Tensor, axis) -> tf.Tensor:
    """
    Returns the shape of the tensor if the axes are removed.

    Parameters
    ----------
    input_tensor: input tensor
    axis: a scalar, list of scalars or 1D Tensor

    Returns
    -------
    A 1D tensor
    """
    axes_range = tf.range(tf.rank(input_tensor), dtype=tf.int32)

    # Prepare axis to 1D tensor
    # Will throw an error if axis is not a scalar or 1D tensor
    if isinstance(axis, tf.Tensor):
        axis = tf.constant(axis, dtype=tf.int32)
    else:
        axis = tf.cast(x=axis, dtype=tf.int32)
    axis = tf.reshape(axis, shape=(-1,))

    # Get actual axis indices if any axis is <0
    neg_axis = tf.less(axis, 0)
    if tf.reduce_any(neg_axis):
        axis = tf.where(condition=neg_axis, x=tf.rank(input_tensor) + axis,
                        y=axis)

    # Create new axis index map and transpose the tensor accordingly
    mask = tf.reduce_all(tf.not_equal(
        tf.reshape(axes_range, shape=(-1, 1)),
        tf.reshape(axis, shape=(1, -1))),
        axis=1)  # boils down to not_in in 1D

    masked_shape = tf.boolean_mask(
        tensor=tf.shape(input_tensor),
        mask=mask)
    return masked_shape


def compute_ranks(values: tf.Tensor,
                  axis: int = -1,
                  direction: str = 'ASCENDING',
                  stable: bool = False) -> tf.Tensor:
    sorted_indices = tf.argsort(values=values, axis=axis,
                                direction=direction,
                                stable=stable)
    ranks = tf.argsort(values=sorted_indices, axis=axis,
                       direction='ASCENDING',
                       stable=False) + 1
    return ranks


def pop_element_from_1d_tensor(input_tensor, index):
    raise NotImplementedError


def insert_element_to_1d_tensor(input_tensor, index):
    raise NotImplementedError


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
