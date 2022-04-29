"""Some stats functions."""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras_utils import utils


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
    q A Tensor containing one or several percentile values of interest
    kwargs kwargs for the tf.stats.percentile function, except 'axis' kwarg. The
        'axis' kwarg is not accepted, this function will flatten the input
        tensor x before computing the percentiles.

    Returns
    -------
    A rank(1) Tensor of length q.
    """
    if keepdims:
        raise NotImplementedError("keepdims=True option is not implemented.")

    if axis is None or (tf.rank(x) == 1 and axis == 0):
        # If axis is None simply flatten the x Tensor and compute percentile
        # over the flattened version
        mask = tf.logical_not(tf.math.is_nan(x))
        # use a boolean mask to remove nans and flatten the Tensor if necessary
        masked_x = tf.boolean_mask(mask=mask, tensor=x, axis=None)
        return tfp.stats.percentile(x=masked_x, q=q, axis=axis,
                                    interpolation=interpolation,
                                    keepdims=False,
                                    validate_args=validate_args,
                                    preserve_gradients=preserve_gradients,
                                    name=name)
    else:
        # Check and process the axis argument
        if not isinstance(axis, tf.Tensor):
            axis = tf.constant(axis, dtype=tf.int32)
        if tf.rank(axis) > 1:
            raise ValueError(
                "Expected a 0 or 1D tensor like object for axis, got"
                " rank." + str(tf.rank(axis).numpy()))
        else:
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
        q_rank = tf.rank(q)
        if tf.equal(q_rank, 0):
            # Expected shape of the Tensor returned by tfp.stats.percentile
            expect_shape = ()
        else:
            expect_shape = (None,)
        res = tf.map_fn(
            fn=lambda t: tfp.stats.percentile(
                x=t, q=q, axis=None,
                interpolation=interpolation,
                keepdims=False,
                validate_args=validate_args,
                preserve_gradients=preserve_gradients,
                name=name),
            elems=ragged_x,
            # enforce return of tf.Tensor instead if tf.RaggedTensor
            fn_output_signature=tf.TensorSpec(shape=expect_shape),
            name="compute_percentile"
        )

        # Reshape vector or 2D result to the correct shape
        restored_dims = init_shape[0:-axis_len]
        if tf.equal(q_rank, 0):
            return tf.reshape(tensor=res, shape=restored_dims)
        elif tf.equal(q_rank, 1):
            res = tf.reshape(tensor=res,
                             shape=tf.concat(
                                 [restored_dims,
                                  tf.slice(tf.shape(q), begin=(0,), size=(1,))],
                                 axis=0))
            # Now move the last axis for different percentiles to the first
            # axis the same way it would have been returned by
            # tfp.stats.percentile
            return utils.move_axis_to_first_dim(x=res, axis=-1)
        else:
            # Should never be reached since tfp.stats.percentile should have
            # thrown an exception already
            raise ValueError("q must be a 0 or 1D Tensor or alike.")
