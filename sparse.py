"""Some useful math operation extensions for Tensorflow."""
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
    indices = tf.range(0, tf.shape(sp_shape)[0], delta=1)
    # FIXME: iteration over tensor values prevents from being a tf.function
    #   should try to use tf.sparse.map_fn and some logical tensor ops to get
    #   rid of if statements
    for i in indices:
        if target_shape[i] is not None:
            if sp_shape[i] == 1 and target_shape[i] > 1:
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