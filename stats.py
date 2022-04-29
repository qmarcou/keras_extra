"""Some stats functions."""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def nanpercentile(x, q, **kwargs) -> tf.Tensor:
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
    if kwargs.get("axis") is not None:
        raise ValueError("nanpercentile does not support the axis argument.")
    mask = tf.logical_not(tf.math.is_nan(x))
    # use a boolean mask to remove nans and flatten the Tensor if necessary
    masked_x = tf.boolean_mask(mask=mask, tensor=x, axis=None)
    return tfp.stats.percentile(masked_x, q, **kwargs)
