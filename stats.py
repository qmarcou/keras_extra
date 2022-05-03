"""Some stats functions."""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras_utils import utils


def _percentile_wrapper(x, q, axis, **kwargs) -> tf.Tensor:
    """A wrapper implementing checks that should be made in tfp.percentile"""
    # Check if any dimension of x is of length 0
    if tf.reduce_any(tf.equal(tf.shape(x), 0)):
        if axis is not None:
            post_compute_shape = utils.shape_without_axis(x, axis)
        else:
            post_compute_shape = tf.constant([], shape=(0,), dtype=tf.int32)
        q = tf.reshape(tensor=q, shape=(-1,))
        return_shape = tf.concat([tf.shape(q), post_compute_shape], axis=0)
        return tf.squeeze(tf.fill(value=np.nan, dims=return_shape))
    else:
        return tfp.stats.percentile(x=x, q=q, axis=axis, **kwargs)


def nanpercentile(x,
                  q,
                  axis=None,
                  interpolation=None,
                  keepdims=False,
                  validate_args=False,
                  preserve_gradients=True,
                  name=None) -> tf.Tensor:
    """
    Compute percentile over all axes of a Tensor ignoring NaNs.
    Parameters
    ----------
    x The Tensor of interest
    q A Tensor or alike containing one or several percentile values of
        interest. Must be 0 or 1D Tensor like.
    kwargs kwargs for the tf.stats.percentile function.

    Returns
    -------
    A Tensor similar to the one returned by tfp.stats.percentile.
    """
    if keepdims:
        raise NotImplementedError("keepdims=True option is not implemented.")

    q = tf.convert_to_tensor(q)
    q_rank = tf.rank(q)
    # Check that q has correct rank
    tf.assert_less(q_rank, 2,
                   message="q must be a 0 or 1D Tensor or alike.")

    if axis is None or (tf.rank(x) == 1 and axis == 0):
        # If axis is None simply flatten the x Tensor and compute percentile
        # over the flattened version
        mask = tf.logical_not(tf.math.is_nan(x))
        # use a boolean mask to remove nans and flatten the Tensor if necessary
        masked_x = tf.boolean_mask(mask=mask, tensor=x, axis=None)
        return _percentile_wrapper(x=masked_x, q=q, axis=axis,
                                   interpolation=interpolation,
                                   keepdims=False,
                                   validate_args=validate_args,
                                   preserve_gradients=preserve_gradients,
                                   name=name)
    else:
        # Check and process the axis argument
        axis = tf.convert_to_tensor(axis, dtype=tf.int32)
        # raise an exception if axis is more than 1D
        tf.assert_less(tf.rank(axis), 2,
                       message="Expected a 0 or 1D tensor like object for "
                               "'axis', got a higher rank (=x) tf.Tensor.")

        axis = tf.reshape(tensor=axis, shape=(-1,))

        # Send the collapsed axes to last dimensions
        x = utils.move_axis_to_last_dim(x=x, axis=axis)

        # Collapse all other dimensions to a vector that can be iterated on
        # the final shape is (-1, length_of_flattened_axis_of_interest)
        init_shape = tf.shape(input=x)
        axis_len = tf.shape(axis)[0]
        axis_collapsed_len = tf.math.cumprod(
            init_shape[-axis_len:]  # axes of interest are last
        )[-1]
        x = tf.reshape(tensor=x, shape=(-1, axis_collapsed_len))

        # Now check nan values and create a ragged Tensor from the boolean mask
        mask = tf.logical_not(tf.math.is_nan(x))
        ragged_x = tf.ragged.boolean_mask(data=x, mask=mask, name='drop_NaNs')

        # Finally apply the percentile function to each row of this ragged
        # Tensor
        res = tf.map_fn(
            fn=lambda t: _percentile_wrapper(
                x=t, q=q, axis=None,
                interpolation=interpolation,
                keepdims=False,
                validate_args=validate_args,
                preserve_gradients=preserve_gradients,
                name=name),
            elems=ragged_x,
            # enforce return of tf.Tensor instead if tf.RaggedTensor
            fn_output_signature=tf.TensorSpec(shape=None, dtype=x.dtype),
            name="compute_percentile"
        )

        # Reshape vector or 2D result to the correct shape
        restored_dims = init_shape[0:-axis_len]
        if tf.equal(q_rank, 0):
            return tf.reshape(tensor=res, shape=restored_dims)
        else:  # rank==1, guaranteed by the assertion on q_rank
            res = tf.reshape(tensor=res,
                             shape=tf.concat(
                                 [restored_dims,
                                  tf.slice(tf.shape(q), begin=(0,),
                                           size=(1,))],
                                 axis=0))
            # Now move the last axis for different percentiles to the first
            # axis the same way it would have been returned by
            # tfp.stats.percentile
            return utils.move_axis_to_first_dim(x=res, axis=-1)
