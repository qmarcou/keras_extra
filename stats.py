"""Some stats functions."""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from . import utils


def _return_nan_percentile(x, q, axis):
    if axis is not None:
        post_compute_shape = utils.shape_without_axis(x, axis)
    else:
        post_compute_shape = tf.constant([], shape=(0,), dtype=tf.int32)
    q = tf.reshape(tensor=q, shape=(-1,))
    return_shape = tf.concat([tf.shape(q), post_compute_shape], axis=0)
    return tf.squeeze(tf.fill(value=np.nan, dims=return_shape))


def _percentile_wrapper(x, q, axis, **kwargs) -> tf.Tensor:
    """A wrapper implementing checks that should be made in tfp.percentile"""
    # Check if any dimension of x is of length 0
    # if tf.reduce_any(tf.equal(tf.shape(x), 0)):
    #     if axis is not None:
    #         post_compute_shape = utils.shape_without_axis(x, axis)
    #     else:
    #         post_compute_shape = tf.constant([], shape=(0,), dtype=tf.int32)
    #     q = tf.reshape(tensor=q, shape=(-1,))
    #     return_shape = tf.concat([tf.shape(q), post_compute_shape], axis=0)
    #     return tf.squeeze(tf.fill(value=np.nan, dims=return_shape))
    # else:
    #     return tfp.stats.percentile(x=x, q=q, axis=axis, **kwargs)

    cond = tf.reduce_any(tf.equal(tf.shape(x), 0))
    return tf.cond(pred=cond,
                   true_fn=lambda: _return_nan_percentile(x=x, q=q, axis=axis),
                   false_fn=lambda: tfp.stats.percentile(x=x, q=q, axis=axis,
                                                         **kwargs))


def nanpercentile(x,
                  q,
                  axis=None,
                  interpolation='linear',
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

    preserve_gradients
    validate_args
    axis
    interpolation ('linear', 'lower', 'higher','midpoint'). Note that 'nearest'
        is NOT supported due to its instability after conversion to effective
        q.
    keepdims
    Note: #FIXME due to the way NaN are handled to enable the use of vectorized
            operations results might not be fully consistent when the number of
             nans is important compared to the total number of values,
             especially when discontinuous interpolations are used

    Returns
    -------
    A Tensor similar to the one returned by tfp.stats.percentile.
    """
    if keepdims:
        raise NotImplementedError("keepdims=True option is not implemented.")

    allowed_interpolations = {'linear', 'lower', 'higher',
                              'midpoint'}

    if interpolation not in allowed_interpolations:
        raise ValueError(
            'Argument `interpolation` must be in {}. Found {}.'.format(
                allowed_interpolations, interpolation))

    # Convert x to tensor if needed
    x = tf.convert_to_tensor(x)

    # enforce float type for later computation on q
    q = tf.convert_to_tensor(q)
    q = tf.cast(x=q, dtype=tf.float64)
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

        # Now check nan values and count them et replace them by +Inf
        mask = tf.math.is_nan(x)
        x = tf.where(condition=mask,
                     x=tf.constant(np.inf, shape=(1, 1)),
                     y=x,
                     name='cast_NaN_to_inf')
        nan_count = tf.reduce_sum(
            tf.cast(x=mask, dtype=tf.int64),
            axis=1,
            keepdims=True  # to obtain a shape(-1,1) Tensor for broadcast
        )
        # Compute the effective percentiles to compute for each row as a Tensor
        # of shape (shape(x)[0],len(q))
        # eff_q  = q*(N_tot-N_nan-1)/(N_tot-1)
        eff_q = tf.multiply(tf.reshape(q, shape=(1, -1)),
                            tf.divide((tf.cast(axis_collapsed_len,
                                               dtype=tf.float64)
                                       - tf.cast(nan_count,
                                                 dtype=tf.float64) - 1.0)
                                      , tf.cast(axis_collapsed_len,
                                                dtype=tf.float64) - 1.0))
        # Flatten it and feed it to the percentile function
        # note that this may not be efficient since it requires to compute
        # many times unuseful percentiles, however this is the only way I found
        # to vectorize the operation, in the end making the operation more
        # efficient, and the sort operation is performed only once anyway
        eff_q = tf.reshape(eff_q, shape=(-1,))
        percentiles = tfp.stats.percentile(
            x=x, q=eff_q, axis=1,
            interpolation=interpolation,
            keepdims=False,
            validate_args=validate_args,
            preserve_gradients=preserve_gradients,
            name=name)
        # Now reshape the result
        percentiles = tf.reshape(
            percentiles,
            shape=(tf.shape(x)[0],
                   tf.shape(tf.reshape(q, shape=(-1,)))[0],
                   tf.shape(x)[0]))
        # move the q axis to the first axis
        percentiles = utils.move_axis_to_first_dim(x=percentiles, axis=1)
        # Now take the diagonal over the last 2 dimensions
        res = tf.linalg.diag_part(percentiles)

        # Finally replace np.inf values introduced by "empty" vectors by nan
        filler = tf.constant(np.nan, shape=(1, 1))
        mask = tf.math.is_inf(res)
        res = tf.where(condition=mask,
                       x=filler,
                       y=res)

        # Reshape vector or 2D result to the correct shape
        restored_dims = init_shape[0:-axis_len]
        if tf.equal(q_rank, 0):
            return tf.reshape(tensor=res, shape=restored_dims)
        else:  # rank==1, guaranteed by the assertion on q_rank
            res = tf.reshape(tensor=res,
                             shape=tf.concat(
                                 [tf.slice(tf.shape(q), begin=(0,),
                                           size=(1,)),
                                  restored_dims
                                  ],
                                 axis=0))

            return res
