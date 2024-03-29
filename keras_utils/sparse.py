"""Some useful math operation extensions for Tensorflow."""
from __future__ import annotations
from typing import List

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
def expand_single_dim(sp_tensor: tf.sparse.SparseTensor,
                      times: int,
                      axis: int):
    out_tensor = sp_tensor
    # sp_tensor_shape = sp_tensor.get_shape()
    # tensor_shape_invariant : tf.TensorShape = tf.TensorShape.concatenate(
    #                             sp_tensor_shape[0:axis],
    #                                    tf.TensorShape([None])).concatenate(
    #                                    sp_tensor_shape[axis+1:],)
    j=tf.constant(0)
    while j<(times-1):
    #for j in tf.range(start=0, limit=times - 1, delta=1):
        # tf.autograph.experimental.set_loop_options(
        #     shape_invariants=[(out_tensor, tf.TensorShape([None]))]
        # )
        # tf.autograph.experimental.set_loop_options(
        #     shape_invariants=[(out_tensor, tensor_shape_invariant)]
        # )
        out_tensor = tf.sparse.concat(axis=axis,
                                      sp_inputs=[out_tensor, sp_tensor])
        j+=1
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
            sp_tensor = expand_single_dim(sp_tensor,
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
def _sparse_reduce_unstsegment_fn(unsorted_segment_fn,
                                  sp_input: tf.SparseTensor,
                                  axis: int | List[int] | tf.Tensor,
                                  keepdims: bool = False,
                                  output_is_sparse=False,
                                  name: str = None) -> tf.SparseTensor:
    """
    math.(unsorted_)segment_sum.

    Args:
        sp: rank 2 sparse tensor
        axis: int, axis along which to sum. Negative indices are not supported
        ordered: if True, other axis indices are assumed to be ascending.

    Returns:
        rank 1 dense tensor equivalent to tf.sparse.reduce_sum(sp, axis=axis)

    TODO:
        - implement gradient:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_grad.py
        cf implementation for unsorted segment max
        https://www.tensorflow.org/guide/create_op#implement_the_gradient_in_python
        https://stackoverflow.com/questions/37924071/tensorflow-writing-an-op-in-python
        https://www.tensorflow.org/api_docs/python/tf/custom_gradient
        - use ops name -> check how to register python ops name
        - implement keepdims
        - check timings
        - update documentation
        - check autographable
    """
    # Adapted from https://github.com/tensorflow/tensorflow/issues/32763
    indices_range = tf.range(0, tf.rank(sp_input))
    if axis is None:
        axis = indices_range

    if not isinstance(axis, tf.Tensor):
        axis = tf.constant(axis)
    axis = tf.reshape(axis, shape=(-1,))  # Enforce 1D tensor
    axis = tf.sort(axis)
    axis, _idx = tf.unique(x=axis)  # make sure values are unique

    # if tf.equal(tf.shape(axis)[0], tf.rank(input=sp_input)):
    #     # Collapse all dimensions, we only need to return a scalar value
    #     return tf.reduce_max(sp_input.values, axis=None, keepdims=False)

    # Get the retained axes based on discarded axes
    mask = tf.reduce_all(tf.not_equal(tf.reshape(indices_range, shape=(-1, 1)),
                                      tf.reshape(axis, shape=(1, -1))),
                         axis=1)
    comp_axes = tf.boolean_mask(indices_range, mask)

    # Extract the corresponding indices
    # this op will discard indices from axis so I'll have to re-insert them
    # if keepdims
    indices, idx = tf.raw_ops.UniqueV2(
        x=tf.gather(params=sp_input.indices, indices=comp_axes, axis=1),
        axis=tf.constant(0, shape=(1,), dtype=tf.int32))

    # I initially thought I could use the sorted of segment_max version with
    # ordered sparseTensors but there were too many corner cases and checks
    # to be performed and would have been limited to reduce in contiguous
    # first dimensions
    values = unsorted_segment_fn(data=sp_input.values,
                                 segment_ids=idx,
                                 num_segments=tf.reduce_max(idx) + 1
                                 )
    if tf.rank(indices) == 1:
        # Indices tensor must be rank 2 to instantiate a SparseTensor
        indices = tf.reshape(tensor=indices, shape=(-1, 1))

    if keepdims:
        # easiest case build sparse tensor with same shape
        raise NotImplementedError("Keepdims option is not implemented yet")
        # Get the new_shape
        new_shape = tf.where(condition=mask,
                             x=sp_input.dense_shape,
                             y=tf.ones(shape=(tf.rank(sp_input)),
                                       dtype=sp_input.dense_shape.dtype))
        # Introduce ones where needed in indices
        # tf.scatter_nd(indices=tf.constant([[0], [2]], shape=(2, 1)),
        #              updates=tf.constant([[2, 2], [3, 3]]), shape=(4, 2))
        indices = tf.transpose(indices)
        indices = tf.tensor_scatter_nd_add(
            tensor=tf.transpose(tf.ones_like(sp_input.indices)),
            indices=comp_axes,
            updates=tf.transpose(indices) - 1)
        indices = tf.transpose(indices)

    else:
        new_shape = tf.gather(params=sp_input.dense_shape,
                              indices=comp_axes)

    sp_out = tf.SparseTensor(values=values, indices=indices,
                             dense_shape=new_shape)
    if output_is_sparse:
        return sp_out
    else:
        return tf.sparse.to_dense(tf.sparse.reorder(sp_out),
                                  default_value=0)


def reduce_max(sp_input: tf.SparseTensor,
               axis: int | List[int] | tf.Tensor,
               keepdims: bool = False,
               output_is_sparse=False,
               name: str = None) -> tf.SparseTensor:
    return _sparse_reduce_unstsegment_fn(
        unsorted_segment_fn=tf.math.unsorted_segment_max,
        sp_input=sp_input,
        axis=axis,
        keepdims=keepdims,
        output_is_sparse=output_is_sparse,
        name=name
    )


def reduce_min(sp_input: tf.SparseTensor,
               axis=None,
               keepdims=None,
               output_is_sparse=False,
               name=None) -> tf.SparseTensor | tf.Tensor:

    return _sparse_reduce_unstsegment_fn(
        unsorted_segment_fn=tf.math.unsorted_segment_min,
        sp_input=sp_input,
        axis=axis,
        keepdims=keepdims,
        output_is_sparse=output_is_sparse,
        name=name
    )
