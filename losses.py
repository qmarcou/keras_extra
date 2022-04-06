from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
import keras.backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
import numpy as np
from scipy.sparse import coo_matrix
import keras_utils.layers


# Some useful ressources:
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
# https://keras.io/getting_started/intro_to_keras_for_researchers/#tracking-losses-created-by-layers
# https://stackoverflow.com/questions/61661739/tensorflow-2-loss-using-hidden-layers-output
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss
# https://stackoverflow.com/questions/50063613/what-is-the-purpose-of-the-add-loss-function-in-keras
# Multiple outputs to use compile loss argument and access y_true:
# https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras


def get_ndim(x: tf.Tensor):
    return tf.shape(x).shape[0]


def weighted_binary_crossentropy_per_node(y_true, y_pred, class_weights,
                                          from_logits=False):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    neg_y_true = tf.ones_like(y_true) - y_true

    # y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels,
    #                             lambda: y_true)

    cross_ent = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    class_weights = tf.cast(class_weights, cross_ent.dtype)
    # Assuming only binary information has been passed in y_true we'll simply
    # reweight each output
    # first reshape the tensors in order to use broadcast
    # BUGFIX: this shape construction fixes issues with Tensors of unknown
    # sizes
    newshape = tf.concat([tf.shape(cross_ent),  # get dynamic shape
                          tf.ones(shape=1, dtype=tf.int32)],  # add 1 dimension
                         axis=0)
    cross_ent = tf.reshape(cross_ent,
                           shape=newshape)

    newshape = tf.concat([
        tf.ones(shape=tf.rank(y_true) - tf.rank(class_weights) + 1
                , dtype=tf.int32),
        tf.shape(class_weights)],  # get dynamic shape
        axis=0)
    class_weights = tf.reshape(class_weights,
                               shape=newshape)
    # Check that class_weights last dimension is either 1 or 2
    tf.assert_less(newshape[-1], tf.constant(3),
                   message="class_weights last dimension must be 1 or 2, only"
                           "binary outcomes are handled by the cross entropy"
                           "function")

    # Perform the product to get a shape = `[batch_size, d0, .. dN, 2]`
    weighted_cross_ent_contrib = tf.multiply(cross_ent, class_weights)

    # Then add the 0 and 1 contribution
    # define slice size to get all values from last dimension
    slice_size = tf.concat([tf.fill(dims=tf.reshape(tf.rank(y_true),
                                                    shape=tf.ones(shape=1,
                                                                  dtype=tf.int32)),
                                    value=-1),
                            tf.ones(shape=1, dtype=tf.int32)],
                           axis=0)
    zeros_y_true_dim = tf.zeros(shape=tf.rank(y_true),
                                dtype=tf.int32)
    # Add contribution for y_true==0
    weighted_cross_ent = tf.multiply(tf.reshape(
        tf.slice(weighted_cross_ent_contrib,
                 begin=tf.concat([zeros_y_true_dim,
                                  tf.zeros(shape=1, dtype=tf.int32)],
                                 axis=0),
                 size=slice_size),
        shape=tf.shape(neg_y_true)),
        neg_y_true)

    # Add contribution for y_true==1
    # Enable broadcast in class_weights are the same for the 2 outcomes
    # output ind is 0 if class_weights are the same, 1 if 2 different weights
    output_ind = tf.multiply(
        tf.ones(shape=1, dtype=tf.int32),
        tf.shape(class_weights)[-1]-1
    )
    weighted_cross_ent += tf.multiply(tf.reshape(
        tf.slice(weighted_cross_ent_contrib,
                 begin=tf.concat([zeros_y_true_dim,
                                  output_ind],
                                 axis=0),
                 size=slice_size),
        shape=tf.shape(y_true)),
        y_true)
    return weighted_cross_ent


def weighted_binary_crossentropy(y_true, y_pred, class_weights,
                                 from_logits=False):
    """Computes a weighted version of the binary crossentropy loss.

    Warnings: Out of simplicity I assume binary labels for y_true (0 or 1), and
    prevent the use of label_smoothing!=0. To fix this I would have to rewrite
    a more complicated version of the binary_crossentropy backend function with
    and without logits.

    Standalone usage:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> weights = [[1.0,10.0],[1.0,5.0]]
    >>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, weights)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.916 , 0.714], dtype=float32)

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
      class_weights: The weights to apply depending on the value of y_true.
       shape = `[d0, .. dN, 2]. If the rank is lower than the expected one
       broadcast will occur on the first dimensions. For instance a shape of
       [2] denotes different weights for true and false outcomes identical for
       every output node and will be interpreted as shape [1,..,1,2].
      from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
        we assume that `y_pred` encodes a probability distribution.

    Returns:
      Weighted binary crossentropy loss value.
      shape = `[batch_size, d0, .. dN-1]`.
    """

    weighted_cross_ent = weighted_binary_crossentropy_per_node(
        y_true=y_true,
        y_pred=y_pred,
        class_weights=class_weights,
        from_logits=from_logits
    )
    return K.mean(weighted_cross_ent, axis=-1)


class WeightedBinaryCrossentropy(LossFunctionWrapper):
    """Computes the Weighted cross-entropy loss between true and predicted labels.

    Use this weighted cross-entropy loss when there are only two label classes
    (assumed to be 0 and 1). For each example, there should be a single
    floating-point value per prediction.


    """

    # TODO : implement per sample class weighting
    #  https://stackoverflow.com/questions/48315094/using-sample-weight-in-keras-for-sequence-labelling
    #  https://github.com/keras-team/keras/issues/2592

    def __init__(self,
                 class_weights,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='binary_crossentropy'):
        """Initializes `BinaryCrossentropy` instance.

        Args:
          from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
              assume that `y_pred` contains probabilities (i.e., values in [0, 1]).
              **Note - Using from_logits=True may be more numerically stable.
          label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0,
            we compute the loss between the predicted labels and a smoothed version
            of the true labels, where the smoothing squeezes the labels towards 0.5.
            Larger values of `label_smoothing` correspond to heavier smoothing.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: (Optional) Name for the op. Defaults to 'binary_crossentropy'.
        """
        super(WeightedBinaryCrossentropy, self).__init__(
            weighted_binary_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            class_weights=tf.constant(class_weights))
        self.from_logits = from_logits


class MCLoss(keras.losses.Loss):
    """
    Implements MCLoss for hierarchical outcomes

    Implements MCLoss proposed in Giunchiglia & Lukasiewicz, NeurIPS 2020,
    "Coherent Hierarchical Multi-Label Classification Networks"
    """

    def __init__(self,
                 adjacency_matrix: np.ndarray | coo_matrix,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='MCLoss',
                 class_weights=None,
                 sparse_adjacency: bool = False,
                 activation='linear'
                 ):
        """

        Parameters
        ----------
        adjacency_matrix: parent->child adjacency matrix (see
            ExtremumConstraintModule "max" documentation for more details)
        from_logits: bool whether binary cross entropy is computed from_logits. Note
            that the trick of using logits on a Dense layer with sigmoid
            will not work because an ECM layer is applied before "cross-entropy"
            computation.
        reduction: Loss reduction type. Defaults to AUTO
        name: str
        class_weights: class weights for the weighted binary cross-entropy
            computation. Defaults to None, meaning equal unit weights for all
            classes.
        sparse_adjacency: bool whether sparse computation should be used for
            the ECM layer.
        activation: str or keras.Activation Activation to be applied in the ECM
            layer. Defaults to 'linear'. 'linear' should be used if from_logits
             is True.
        """
        super(MCLoss, self).__init__(
            name=name,
            reduction=reduction,
        )
        if class_weights is None:
            self._class_weights = tf.constant([1.0])
        else:
            self._class_weights = class_weights
        self._from_logits = from_logits
        self._MCMact = keras_utils.layers.ExtremumConstraintModule(
            activation=activation,
            extremum='max',
            adjacency_matrix=adjacency_matrix,
            sparse_adjacency=sparse_adjacency
        )

    def call(self, y_true, y_pred):
        """"""
        if tf.is_tensor(y_pred) and tf.is_tensor(y_true):
            y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
                y_pred, y_true)
        # Cast y_true to the correct type for further computations
        y_true = tf.cast(y_true, y_pred.dtype)

        # Compute positive examples custom cross-ent contributions
        y_pred_true = tf.multiply(y_true, y_pred)
        pos_mcm = self._MCMact(y_pred_true)
        pos_mcloss = weighted_binary_crossentropy_per_node(
            y_true=y_true,
            y_pred=pos_mcm,
            class_weights=self._class_weights,
            from_logits=self._from_logits)
        # Now cast y_true to the correct type for cross-ent multiplication
        y_true = tf.cast(y_true, pos_mcloss.dtype)
        pos_mcloss = tf.multiply(y_true, pos_mcloss)
        # Compute negative examples custom cross-ent contributions
        neg_mcm = self._MCMact(y_pred)
        neg_mcloss = weighted_binary_crossentropy_per_node(
            y_true=y_true,
            y_pred=neg_mcm,
            class_weights=self._class_weights,
            from_logits=self._from_logits)
        neg_y_true = tf.ones_like(y_true) - y_true
        neg_mcloss = tf.multiply(neg_y_true, neg_mcloss)
        # Sum the two parts
        mcloss = tf.add(pos_mcloss, neg_mcloss)

        # ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
        return K.mean(mcloss, axis=-1)

    # TODO: create serializing and reading functions
    # def get_config(self):
    #     config = {}
    #     for k, v in six.iteritems(self._fn_kwargs):
    #         config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    #     base_config = super(LossFunctionWrapper, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


class TreeMinLoss(keras.losses.Loss):
    """
    Implements TreeMin loss for hierarchical outcomes

    Implements TreeMin loss proposed in Li, Zhou, Wang, Li, Yang Preprint 2022,
    "Deep Hierarchical Semantic Segmentation"
    """

    def __init__(self,
                 adjacency_matrix: np.ndarray | coo_matrix,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='TreeMinLoss',
                 class_weights=None,
                 sparse_adjacency: bool = False,
                 activation='linear'
                 ):
        """

        Parameters
        ----------
        adjacency_matrix: parent->child adjacency matrix (see
            ExtremumConstraintModule "max" documentation for more details)
        from_logits: bool whether binary cross entropy is computed from_logits. Note
            that the trick of using logits on a Dense layer with sigmoid
            will not work because an ECM layer is applied before "cross-entropy"
            computation.
        reduction: Loss reduction type. Defaults to AUTO
        name: str
        class_weights: class weights for the weighted binary cross-entropy
            computation. Defaults to None, meaning equal unit weights for all
            classes.
        sparse_adjacency: bool whether sparse computation should be used for
            the ECM layer.
        activation: str or keras.Activation Activation to be applied in the ECM
            layer. Defaults to 'linear'. 'linear' should be used if from_logits
             is True.
        """
        super(MCLoss, self).__init__(
            name=name,
            reduction=reduction,
        )
        if class_weights is None:
            self._class_weights = tf.constant([1.0])
        else:
            self._class_weights = class_weights
        self._from_logits = from_logits
        self._MaxCMact = keras_utils.layers.ExtremumConstraintModule(
            activation=activation,
            extremum='max',
            adjacency_matrix=adjacency_matrix,
            sparse_adjacency=sparse_adjacency
        )
        self._MinCMact = keras_utils.layers.ExtremumConstraintModule(
            activation=activation,
            extremum='min',
            adjacency_matrix=adjacency_matrix.transpose(),
            sparse_adjacency=sparse_adjacency
        )

    def call(self, y_true, y_pred):
        """"""
        if tf.is_tensor(y_pred) and tf.is_tensor(y_true):
            y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
                y_pred, y_true)
        # Cast y_true to the correct type for further computations
        y_true = tf.cast(y_true, y_pred.dtype)

        # Compute positive examples custom cross-ent contributions
        #y_pred_true = tf.multiply(y_true, y_pred)
        pos_mcm = self._MinCMact(y_pred)
        pos_mcloss = weighted_binary_crossentropy_per_node(
            y_true=y_true,
            y_pred=pos_mcm,
            class_weights=self._class_weights,
            from_logits=self._from_logits)
        # Now cast y_true to the correct type for cross-ent multiplication
        y_true = tf.cast(y_true, pos_mcloss.dtype)
        pos_mcloss = tf.multiply(y_true, pos_mcloss)
        # Compute negative examples custom cross-ent contributions
        neg_mcm = self._MaxCMact(y_pred)
        neg_mcloss = weighted_binary_crossentropy_per_node(
            y_true=y_true,
            y_pred=neg_mcm,
            class_weights=self._class_weights,
            from_logits=self._from_logits)
        neg_y_true = tf.ones_like(y_true) - y_true
        neg_mcloss = tf.multiply(neg_y_true, neg_mcloss)
        # Sum the two parts
        mcloss = tf.add(pos_mcloss, neg_mcloss)

        # ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
        return K.mean(mcloss, axis=-1)

    # TODO: create serializing and reading functions
    # def get_config(self):
    #     config = {}
    #     for k, v in six.iteritems(self._fn_kwargs):
    #         config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    #     base_config = super(LossFunctionWrapper, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))