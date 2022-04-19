"""Some useful math operation extensions for Tensorflow."""
from __future__ import annotations

import tensorflow as tf


# Solutions based on the following links
# https://github.com/tensorflow/tensorflow/issues/1241
# https://github.com/tensorflow/tensorflow/issues/10003
# https://www.tensorflow.org/api_docs/python/tf/sparse/map_values
# https://stackoverflow.com/questions/39650169/sparsetensor-equivalent-of-tf-tile
# https://stackoverflow.com/questions/48697079/tensorflow-batch-sparse-multiply
# https://github.com/tensorflow/tensorflow/issues/38429
# https://github.com/tensorflow/tensorflow/issues/33336

@tf.function
def expend_single_dim(sp_tensor: tf.sparse.SparseTensor,
                      times: int,
                      axis: int):
    out_tensor = sp_tensor
    for j in tf.range(start=0, limit=times - 1, delta=1):
        out_tensor = tf.sparse.concat(axis=axis,
                                      sp_inputs=[out_tensor, sp_tensor])
    return out_tensor


# @tf.function
def expend_unit_dim(sp_tensor: tf.SparseTensor,
                    target_shape: tf.TensorShape) -> tf.SparseTensor:
    # FIXME find a way to make it work with autograph
    if isinstance(target_shape, tf.TensorShape) and \
            not target_shape.is_fully_defined():
        return sp_tensor
    sp_shape = sp_tensor.dense_shape
    sp_rank = tf.rank(sp_tensor)
    mask: tf.Tensor = tf.logical_and(tf.equal(sp_shape, 1),
                                     tf.greater(target_shape, 1))
    indices = tf.where(condition=mask)
    indices = tf.squeeze(indices, axis=1)  # check that this will always work
    for i in tf.range(0, sp_rank, delta=1):
        if mask[i]:
            sp_tensor = expend_single_dim(sp_tensor,
                                          times=target_shape[i],
                                          axis=int(i))
    return sp_tensor


# @tf.function
def sparse_dense_multiply(sparse_t: tf.SparseTensor,
                          dense_t: tf.Tensor,
                          keep_sparse=True) -> tf.SparseTensor:
    """
    Performs  element wise multiplication with bidirectional fake broadcast.

    As of tf v2.8.0 sparse tensor multiplication via __mul__ can only handle
    broadcast of the dense Tensor dimensions, not the sparse one.
    This function handles this case via fake broadcast, meaning through
    duplication of data in the broadcasted axis if keep_sparse==True.
    If keep_sparse==False the SparseTensor is converted to a dense one to use
    natural Tensor broadcast.
    Caveat: as for __mul__ if dense contains NaN values results might be
     inconsistent

    Parameters
    ----------
    sparse_t
    dense_t
    keep_sparse

    Returns
    -------

    """
    # Assert compatible shapes
    if keep_sparse:
        sparse_t = expend_unit_dim(sparse_t, tf.shape(dense_t))
        return sparse_t.__mul__(dense_t)
    else:
        sparse_t = tf.sparse.to_dense(sparse_t)
        return tf.sparse.from_dense(tf.multiply(sparse_t, dense_t))


# @tf.function
def reduce_max_single_axis(sp_input: tf.SparseTensor,
                           axis: int,
                           keepdims=None,
                           ordered=False,
                           name=None) -> tf.SparseTensor:
    """
    math.(unsorted_)segment_sum.

    Args:
        sp: rank 2 sparse tensor
        axis: int, axis along which to sum
        ordered: if True, other axis indices are assumed to be ascending.

    Returns:
        rank 1 dense tensor equivalent to tf.sparse.reduce_sum(sp, axis=axis)
    """
    if axis is None:
        axis = tf.range(0, tf.rank(sp_input))
    axis = tf.constant(axis)
    axis = tf.reshape(axis, shape=(-1,))  # Enforce 1D tensor
    axis, _idx = tf.unique(x=axis)  # make sure values are unique

    if tf.shape(axis)[0] == tf.rank(input=sp_input):
        # Collapse all dimensions, we only need to return a scalar value
        return tf.reduce_max(sp_input.values, axis=None, keepdims=False)

    # Get the retained axes based on discarded axes
    indices_range = tf.range(0, tf.rank(sp_input))
    mask = tf.reduce_any(tf.not_equal(tf.reshape(indices_range, shape=(-1, 1)),
                                      tf.reshape(axis, shape=(1, -1))),
                         axis=1)
    comp_axes = tf.boolean_mask(indices_range, mask)

    # Adapted from https://github.com/tensorflow/tensorflow/issues/32763

    # Extract the corresponding indices
    # this op will discard indices from axis so I'll have to re-insert them
    # if keepdims
    indices, idx = tf.raw_ops.UniqueV2(
        x=tf.gather(params=sp_input.indices, indices=comp_axes, axis=1),
        axis=tf.constant(0, shape=(1,), dtype=tf.int32))
    # requires a 1D axis tensor

    if ordered:
        values = tf.math.segment_max(data=sp_input.values,
                                     segment_ids=idx)
    else:
        values = tf.math.unsorted_segment_max(data=sp_input.values,
                                              segment_ids=idx,
                                              num_segments=tf.reduce_max(idx)+1
                                              )
    if tf.rank(indices) == 1:
        # Indices tensor must be rank 2 to instantiate a SparseTensor
        indices = tf.reshape(tensor=indices, shape=(-1, 1))

    if keepdims:
        # easiest case build sparse tensor with same shape
        raise NotImplementedError("Keepdims option is not implemented yet")
        return tf.SparseTensor()
    else:
        # "slice" indices and shape tensor to exclude single dimension (use
        # tf.gather with tf.where(tf.not_equal(range(0,rank),axis))
        new_shape = tf.gather(params=sp_input.dense_shape,
                              indices=comp_axes)
        # print(comp_axes)
        # print(new_shape)
        # print(indices)
        # print(values)
        return tf.SparseTensor(values=values, indices=indices,
                               dense_shape=new_shape)


def reduce_min(sp_input: tf.SparseTensor,
               axis=None,
               keepdims=None,
               output_is_sparse=False,
               name=None) -> tf.SparseTensor | tf.Tensor:
    sp_neg = tf.sparse.map_values(tf.math.negative, sp_input)
    sp_min = tf.sparse.reduce_max(sp_neg, axis=axis, keepdims=keepdims,
                                  output_is_sparse=True,
                                  name=name)
    res = tf.sparse.map_values(tf.math.negative, sp_min)
    if output_is_sparse:
        return res
    else:
        return tf.sparse.to_dense(res)
