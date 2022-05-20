"""A collection of custom keras layers."""
from __future__ import annotations
import numpy as np
from numpy import dtype, clip
from scipy.sparse import coo_matrix, isspmatrix_coo, isspmatrix
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation
from keras import activations
from tensorflow.keras import backend as K
from typing import Optional
from enum import Enum
import keras_utils.sparse


# Useful references:
# https://www.tensorflow.org/guide/keras/custom_layers_and_models

class Extremum(Enum):
    Max = "max",
    Min = "min"


class ExtremumConstraintModule(Activation):
    """
    Implements a Max Constraint Module activation.

    Implements the MCM activation described in Giunchiglia & Lukasiewicz,
    "Coherent Hierarchical Multi-Label Classification Networks", NeurIPS 2020.
    The MCM activation takes an arbitrary activation function (though described
    with sigmoid/logistic in the paper) and uses the hierarchy information
    provided by an adjacency matrix to compute a resulting activation imposing
    a coherent hierarchical constraint.
    This ECM class allows to use either min or max to impose the constraint.
    Using min or max result in two different strategies:
    - min: the expected adjacency matrix gives [child,parent] relationship,
        the hierarchical activation is made coherent by setting the activation
        of a node to the minimum of its ancestors activation. Such a setting
        is useful if we expect to trust more the parent classifier, and is
        closer to the "local" classifier approach.
    - max: the expected adjacency matrix gives [parent,child] relationship,
        the hierarchical classification is made coherent by a "roll-up"
        approach in which information about children nodes is used to compute
        the resulting parent activation. This is the strategy proposed in the
        MCM paper.
    sparse_adjacency: works only if there's no need to backpropagate gradients
    """

    def __init__(self, activation, extremum: str,
                 adjacency_matrix: np.ndarray | coo_matrix,
                 sparse_adjacency: bool = False,
                 **kwargs):
        super(Activation, self).__init__(trainable=False, **kwargs)
        # self.supports_masking = True
        # self.trainable = False
        self.activation = activations.get(activation)
        self.sparse_adjacency = sparse_adjacency
        self._act_new_shape = None
        self._filtered_act_template = None

        # Check that provided matrix is square
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]

        # Treat adjacency matrix as sparse if needed
        if self.sparse_adjacency:
            if isspmatrix(adjacency_matrix):
                if not isspmatrix_coo(adjacency_matrix):
                    adjacency_matrix = adjacency_matrix.tocoo()
            else:
                adjacency_matrix = coo_matrix(adjacency_matrix)
            # Create a sparse Tensor from the hierarchy information
            self.adjacency_mat = tf.SparseTensor(
                indices=np.mat([adjacency_matrix.row,
                                adjacency_matrix.col])
                    .transpose(),
                values=tf.cast(adjacency_matrix.data, dtype=self.dtype),
                dense_shape=adjacency_matrix.shape)

            # reorder in row major as advised in tf docs
            self.adjacency_mat = tf.sparse.reorder(self.adjacency_mat)
            # add ones on the diagonal
            self.adjacency_mat = tf.sparse.add(self.adjacency_mat,
                                               tf.sparse.eye(
                                                   num_rows=
                                                   adjacency_matrix.shape[0],
                                                   num_columns=
                                                   adjacency_matrix.shape[0],
                                                   dtype=self.dtype))
            # TODO
            # check that all values are 1s
        else:
            self.adjacency_mat = tf.constant(adjacency_matrix,
                                             dtype=self.dtype)
            # Add ones on the diagonal to include considered class predictions
            if all(tf.equal(tf.linalg.diag_part(self.adjacency_mat), 0.0)):
                self.adjacency_mat = tf.add(self.adjacency_mat,
                                            tf.eye(num_rows=adjacency_matrix
                                                   .shape[0],
                                                   num_columns=adjacency_matrix
                                                   .shape[0],
                                                   dtype=self.dtype))
            elif all(tf.equal(tf.linalg.diag_part(self.adjacency_mat), 0.0)):
                pass
            else:
                raise ValueError("The diagonal of the adjacency matrix must "
                                 "be either all 1s or all 0s.")
            # Check that the provided adjacency matrix makes sense
            mask = tf.logical_or(
                tf.equal(self.adjacency_mat, 0.0),
                tf.equal(self.adjacency_mat, 1.0))
            if not tf.reduce_all(mask):
                raise ValueError("Invalid adjacency matrix: the adjacency "
                                 "matrix should only contain 0s and 1s. The "
                                 "passed AM contains the following unique "
                                 "values:"
                                 + str(tf.unique(tf.reshape(self.adjacency_mat,
                                                            shape=(-1,)))))

        extremum = str(extremum).lower()
        if extremum in ['min', 'minimum']:
            self.extremum = Extremum.Min
        elif extremum in ['max', 'maximum']:
            self.extremum = Extremum.Max
        else:
            raise ValueError("Invalid 'extremum' argument.")

        if sparse_adjacency:
            self._select_func = self._select_sparse
            # Create a placeholders for a working copy of the adjacency matrix
            # that can be reshaped according to batch size
            # Since SparseTensors cannot be used as tf.Variables or other TF
            # mutable types, I have to define the different components of the
            # sparseTensor (values, indices, shape) as variables and
            # re-instantiate a sparseTensor everytime
            # Values is the only placeholder that can be instantiated here as
            # it must be a vector
            self._adj_mat_wc_values = tf.Variable(self.adjacency_mat.values,
                                                  shape=(None,),
                                                  trainable=False)
            self._adj_mat_wc_indices = None
            self._adj_mat_wc_shape = None
            if self.extremum == Extremum.Min:
                self._extremum_func = keras_utils.sparse.reduce_min
            else:
                # Use custom reduce_max since tf.sparse.reduce_max has no
                # gradient implemented anyway and my implementation is ~3x
                # faster
                self._extremum_func = keras_utils.sparse.reduce_max
        else:
            self._select_func = self._select_dense
            if self.extremum == Extremum.Min:
                self._extremum_func = tf.reduce_min
            else:
                self._extremum_func = tf.reduce_max

    def call(self, inputs, *args, **kwargs):
        # Compute raw activations
        act = self.activation(inputs)
        act = tf.reshape(act, shape=self._act_new_shape, name='act_reshape')

        # TODO create an Adj mat working copy for sparse
        # _select_func function:
        # if not sparse:
        # Select values from the hierarchy and add them to the template filled
        # with -np.inf (if max) or np.inf (min)
        # For a single layer network with large hierarchy (1300 items)
        # using tf.where is 1.5 times slower than tf.multiply (tf.where took
        # 22.6% of computation time, tf.multiply 15.2%)
        # if sparse:
        # use a sparse dense multiplication between activation tensor and
        # the adjacency matrix. The sparse.reduce_xxx functions overlook
        # implicit 0s in their computation so there is no need to use np.inf
        if self.sparse_adjacency:
            adj_mat = tf.cond(
                self._is_adj_mat_exp_correct_size(act),
                true_fn=self._get_adj_mat_exp,
                false_fn=lambda: self._update_adj_mat_exp_first_dim(act),
                name='cond_update_wc_adj_mat'
            )

            hier_act = self._select_func(act,
                                         adj_mat=adj_mat)
        else:
            hier_act = self._select_func(act,
                                         adj_mat=self.adjacency_mat)

        extr_act = self._extremum_func(
            hier_act,
            # reduce last axis but neg indices not supported in
            # keras_utils.sparse.reduce_max
            axis=tf.rank(hier_act) - 1,
            keepdims=False,
            name="ecm_collapse")

        # Bug Fix: This reshape is introduced to fix a weird bug at graph
        # creation time in evaluate and fit (keras). Code to reproduce the bug:
        # ecm_layer_max = layers.ExtremumConstraintModule(
        #     activation="linear",
        #     extremum="max",
        #     adjacency_matrix=adj_mat.transpose(),
        #     sparse_adjacency=True)
        #
        # model = keras.Sequential([keras.layers.Input(shape=(4,)),
        # ecm_layer_max])
        # model.compile(loss=keras.losses.binary_crossentropy, metrics=[
        # metrics.RankErrorsAtPercentile(q=50, no_true_label_value=1.0,
        # interpolation='linear')], )
        # model.fit(args)
        # would raise: "ValueError: Number of mask dimensions must be specified,
        # even if some dimensions are None.  E.g. shape=[None] is ok,
        # but shape=None is not." from the boolean_mask function called in
        # nanpercentile.
        # I have the impression that upon converting a sparse Tensor to dense
        # does not give the correct shape signature at graph construction, and
        # by introducing this reshape somewhat fixes the issue, but it's hacky.
        # Everything was running fine on eager execution
        return tf.reshape(tensor=extr_act, shape=tf.shape(inputs))

    def build(self, input_shape: tf.TensorShape):
        # reshape with length 1 first dimension for broadcasting over
        # sample and possibly other dimensions
        # input_shape does not contain the explicit batch size dimension
        # but only None
        new_shape = tf.concat([tf.ones(shape=tf.maximum(input_shape.rank - 1,
                                                        1),
                                       dtype=tf.int32),
                               tf.fill(dims=2, value=input_shape[-1])],
                              axis=0)

        if self.sparse_adjacency:
            # if this line throws an error it means the adjacency matrix does
            # not have the correct shape
            self.adjacency_mat = tf.sparse.reshape(self.adjacency_mat,
                                                   shape=new_shape)
        else:
            self.adjacency_mat = tf.cast(self.adjacency_mat, dtype=tf.bool)
            # if this line throws an error it means the adjacency matrix does
            # not have the correct shape
            self.adjacency_mat = tf.reshape(self.adjacency_mat,
                                            shape=new_shape)

        # Compute the shape required for the activation tensor
        # Add a broadcasting dimension to the activation tensor
        self._act_new_shape = tf.concat([
            tf.constant([-1], dtype=tf.int32),  # batch size
            input_shape[1:-1],
            tf.ones(shape=1, dtype=tf.int32),  # broadcasting dim
            input_shape[-1:]],  # output dimension
            axis=0)

        if self.sparse_adjacency:
            # Expand sparse adj mat unit dimensions to the ones of activations
            # This will duplicate values in the adjacency matrix, but this is
            # necessary since tf.SparseTensor.__mul__ will only broadcast dense
            # dimensions. However, since the batch_size may vary I cannot use
            # __mul__ directly anyway
            self.adjacency_mat = keras_utils.sparse.expend_unit_dim(
                self.adjacency_mat,
                self._act_new_shape)
            # Initialise the variable adj mat placeholder
            # As the number of non zero values/indices may vary the first
            # dimension must be None so as to allow to assign different length
            # values/indices
            self._adj_mat_wc_shape = tf.Variable(
                initial_value=self.adjacency_mat.dense_shape,
                shape=tf.rank(self.adjacency_mat),
                trainable=False,
                dtype=tf.int64
            )
            self._adj_mat_wc_indices = tf.Variable(
                initial_value=self.adjacency_mat.indices,
                shape=(None, tf.rank(self.adjacency_mat)),
                trainable=False
            )
            # update values initialized in __init__
            self._adj_mat_wc_values.assign(self.adjacency_mat.values)

        # Create a base tensor to be filled with values
        if not self.sparse_adjacency:
            act_template_shape = tf.ones(shape=input_shape.rank + 1,
                                         dtype=tf.int32)
            if self.extremum == Extremum.Min:
                self._filtered_act_template = tf.fill(value=np.Inf,
                                                      dims=act_template_shape)
            elif self.extremum == Extremum.Max:
                self._filtered_act_template = tf.fill(value=-np.Inf,
                                                      dims=act_template_shape)

    def get_config(self):
        config = {'extremum': str(self.extremum),
                  'sparse': self.sparse_adjacency}
        # FIXME reshape adj_mat to output it
        base_config = super(ExtremumConstraintModule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _is_adj_mat_exp_correct_size(self, act):
        act_shape = tf.shape(act, out_type=tf.int64)
        return tf.squeeze(tf.slice(tf.equal(act_shape, self._adj_mat_wc_shape),
                                   begin=(0,),
                                   size=(1,)))

    def _update_adj_mat_exp_first_dim(self, act):
        sparse_template = keras_utils.sparse.expend_single_dim(
            self.adjacency_mat,
            axis=0,
            times=tf.squeeze(tf.slice(tf.shape(act), begin=(0,), size=(1,))))
        self._adj_mat_wc_shape.assign(sparse_template.dense_shape)
        self._adj_mat_wc_values.assign(sparse_template.values)
        self._adj_mat_wc_indices.assign(sparse_template.indices)
        return self._get_adj_mat_exp()

    def _get_adj_mat_exp(self):
        return tf.SparseTensor(values=self._adj_mat_wc_values,
                               indices=self._adj_mat_wc_indices,
                               dense_shape=self._adj_mat_wc_shape)

    def _select_dense(self, act, adj_mat):
        return tf.where(condition=adj_mat,
                        x=act,
                        y=self._filtered_act_template,
                        name="select_hier")

    def _select_sparse(self, act, adj_mat):
        return adj_mat.__mul__(act)

    def is_input_hier_coherent(self, inputs, collapse_all=True) -> bool:
        ecm_inputs = self(inputs)
        if collapse_all:
            return tf.reduce_all(tf.equal(inputs, ecm_inputs))
        else:
            # Only collapse the last dimension
            return tf.reduce_all(tf.equal(inputs, ecm_inputs), axis=-1)

class DenseHierL2Reg(keras.layers.Dense):
    """
    A Dense layer with regularization loss based on hierarchical relationships.

    This dense layer embeds a regularization term imposing similar weights
    for parent and child nodes based on the L2 norm of the difference between
    child and parent weights.

    This regularization can be made based on input or output relationship.
    The output regularization is the implementation of Gopal and Yang,
    "Recursive Regularization for Large-scale Classification with Hierarchical
    and Graphical Dependencies",2013. As pointed out in the latter paper can
    also be applied in more general graph relationships.

    Note: this class should be expanded to accept several hierarchies as input
    to enable multi-graph regularization with different weights for each graph.
    """

    def __init__(self,
                 units,
                 adjacency_matrix: np.ndarray | coo_matrix,
                 hier_side: str,
                 regularization_factor: float = 0.01,
                 tree_like: bool = False,
                 **kwargs):
        """

        Parameters
        ----------
        units   Number of units of the Dense layer
        adjacency_matrix    Adjacency matrix, must be a square matrix
        containing 0s and 1s, its size must match the input or output (number
        of units) size. The orientation of the links (child->parent or
        parent->child) doesn't matter, however you should not provide redundant
         links.
        hier_side: 'out/'output' or 'in'/'input' depending whether the
        hierarchical relationships describe inputs our outputs
        regularization_factor: magnitude of the regularization term, defaults
        to .001.
        tree_like: whether the adjacency matrix is tree like and should be
        checked as being tree like.
        kwargs: other keyword arguments passed to the Dense Layer constructor
        """
        super(DenseHierL2Reg, self).__init__(units=units, **kwargs)

        # Check the adjacency matrix general logic
        _check_dense_adj_mat(adjacency_matrix)

        # Check that the adjacency matrix is a tree
        # If tree like there must be a direction in which the matrix sums to 1s
        # and 0s since nodes can have at most 1 parent
        if tree_like:
            adj_sum = tf.reduce_sum(adjacency_matrix, axis=0)
            first_ax_pred = tf.logical_not(tf.reduce_all(tf.logical_or(
                    tf.equal(adj_sum, tf.constant(1, dtype=adj_sum.dtype)),
                    tf.equal(adj_sum, tf.constant(0, dtype=adj_sum.dtype)))))

            # check the second axis
            adj_sum = tf.reduce_sum(adjacency_matrix, axis=1)
            sec_ax_pred = tf.logical_not(tf.reduce_all(tf.logical_or(
                    tf.equal(adj_sum, tf.constant(1, dtype=adj_sum.dtype)),
                    tf.equal(adj_sum, tf.constant(0, dtype=adj_sum.dtype)))))
            if tf.logical_and(first_ax_pred, sec_ax_pred):
                print(adj_sum)
                raise ValueError("The adjacency matrix is not tree like.")

        # Make the matrix an adjacency list 2D (L,2) Tensor
        self.adj_list = tf.where(condition=adjacency_matrix)

        # Store adj_mat size for further checks in build
        self._adj_mat_size = tf.shape(adjacency_matrix)[0]

        hier_side = str(hier_side).lower()
        if hier_side in ['in', 'input']:
            self.hier_side = "input"
            self.weights_vec_axis = 0
            self.weights_concat_axis = 1
        elif hier_side in ['out', 'output']:
            self.hier_side = "output"
            self.weights_vec_axis = 1
            self.weights_concat_axis = 0
        else:
            raise ValueError("Invalid 'hier_side' argument.")

        if regularization_factor <= 0:
            raise ValueError("Regularization factor must be > 0")
        self.regularization_factor = tf.constant(regularization_factor,
                                                 shape=())

    def build(self, input_shape: tf.TensorShape):
        super(DenseHierL2Reg, self).build(input_shape=input_shape)

        if tf.shape(self.kernel)[self.weights_vec_axis] != self._adj_mat_size:
            raise ValueError("The size of the adjacency matrix and weights "
                             "dimension do not match.")

    def call(self, inputs):
        # call a tf.func computing the L2 norm of difference with each parent
        # concat weights and bias to a general weight tensor
        if self.hier_side == "output":
            concat_weights = tf.concat(values=[self.kernel,
                                               tf.reshape(self.bias,
                                                          shape=(1, -1))],
                                       axis=self.weights_concat_axis,
                                       name="concatWeightsBias")
        else:
            concat_weights = self.kernel

        # Add the L2 norms of the difference between parent/child vectors
        self.add_loss(self.regularization_factor *
                      tf.reduce_sum(tf.square(
                          _dense_compute_hier_weight_diff_tensor(
                              weights=concat_weights,
                              adj_list=self.adj_list,
                              axis=self.weights_vec_axis
                          )
                      )))
        return super(DenseHierL2Reg, self).call(inputs=inputs)


@tf.function
def _dense_compute_hier_weight_diff_tensor(weights: tf.Tensor,
                                           adj_list: tf.Tensor,
                                           axis: int) -> tf.Tensor:
    x = tf.gather(params=weights, indices=adj_list[:, 0], axis=axis,
                  name="getx_vectors")
    y = tf.gather(params=weights, indices=adj_list[:, 1], axis=axis,
                  name="gety_vectors")
    sub = tf.subtract(x=x, y=y, name="subtractxy")

    if axis != 0:
        # Transpose the obtained tensor such that the first dimension
        # correspond to the different entries in the adjacency_list
        indices_range = tf.range(0, tf.rank(weights))
        if axis < 0:
            axis = indices_range[axis]

        if isinstance(axis, tf.Tensor):
            axis = tf.reshape(axis, shape=(1,))
        else:
            axis = tf.constant(axis, shape=(1,))

        mask = tf.not_equal(indices_range,
                            axis)
        comp_axes = tf.boolean_mask(indices_range, mask)

        return tf.transpose(a=sub, perm=tf.concat([axis, comp_axes], axis=0))
    else:
        return sub


def _check_dense_adj_mat(adj_mat: tf.Tensor) -> None:
    adjacency_mat = tf.constant(adj_mat,  # cast to a general type
                                dtype=tf.float64)
    # Check dimensions
    if tf.rank(adjacency_mat) != 2:
        raise ValueError("The adjacency matrix must be a 2D matrix")

    adj_shape = tf.shape(adjacency_mat)
    if adj_shape[0] != adj_shape[1]:
        raise ValueError("The adjacency matrix must be a square matrix")

    # Check that the provided adjacency matrix makes sense
    mask = tf.logical_or(
        tf.equal(adjacency_mat, 0.0),
        tf.equal(adjacency_mat, 1.0))
    if not tf.reduce_all(mask):
        raise ValueError("Invalid adjacency matrix: the adjacency "
                         "matrix should only contain 0s and 1s. The "
                         "passed AM contains the following unique "
                         "values:"
                         + str(tf.unique(tf.reshape(adjacency_mat,
                                                    shape=(-1,)))))
