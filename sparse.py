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


# @tf.function
def expend_unit_dim(sp_tensor: tf.SparseTensor,
                    target_shape: tf.TensorShape) -> tf.SparseTensor:
    sp_shape = tf.shape(sp_tensor)
    mask = tf.logical_and(tf.equal(sp_shape, 1), tf.greater(target_shape, 1))
    indices = tf.where(condition=mask)
    indices = tf.squeeze(indices, axis=1)  # check that this will always work
    for i in indices:
        sp_tensor = tf.sparse.concat(axis=i,
                                     sp_inputs=[sp_tensor for j in
                                                tf.range(0,
                                                         target_shape[
                                                             i])])
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
        sparse_t = expend_unit_dim(sparse_t, dense_t.shape)
        return sparse_t.__mul__(dense_t)
    else:
        sparse_t = tf.sparse.to_dense(sparse_t)
        return tf.sparse.from_dense(tf.multiply(sparse_t, dense_t))


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
