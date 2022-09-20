"""A collection of custom keras layers."""
from __future__ import annotations
import numpy as np
from numpy import dtype, clip
from scipy.constants import value
from scipy.sparse import coo_matrix, isspmatrix_coo, isspmatrix
from scipy.sparse import identity as sci_sparse_identity
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation
from keras import activations
from tensorflow.keras import backend as K
from typing import Optional
from enum import Enum
import keras_utils.sparse
import keras_utils.utils
from keras_utils.callbacks import DataEvaluator


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
    """

    def __init__(self, activation, extremum: str,
                 adjacency_matrix: np.ndarray | coo_matrix | tf.Tensor |
                                   tf.SparseTensor,
                 **kwargs):
        super(Activation, self).__init__(trainable=False, **kwargs)
        # self.supports_masking = True
        # self.trainable = False
        self.activation = activations.get(activation)

        # Convert input adjacency matrix to scipy COO matrix
        if (not isinstance(adjacency_matrix, tf.SparseTensor) and
                not isspmatrix(adjacency_matrix)):
            if isinstance(adjacency_matrix, tf.Tensor):
                adjacency_matrix = adjacency_matrix.numpy()
            adjacency_matrix = np.asmatrix(adjacency_matrix)
            adjacency_matrix = coo_matrix(adjacency_matrix)
        elif isinstance(adjacency_matrix, tf.SparseTensor):
            adjacency_matrix = coo_matrix((adjacency_matrix.values,
                                           (adjacency_matrix.indices[:, 0],
                                            adjacency_matrix.indices[:, 1])),
                                          shape=tf.shape(adjacency_matrix))
        else:
            if not isspmatrix_coo(adjacency_matrix):
                adjacency_matrix = adjacency_matrix.tocoo()

        # Check that the provided matrix is square
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]

        # Check that all values are 0 or 1
        if not np.logical_or(
                adjacency_matrix.data == 1,
                adjacency_matrix.data == 0).all():
            raise ValueError("Invalid adjacency matrix: the adjacency "
                             "matrix should only contain 0s and 1s. The "
                             "passed AM contains the following unique non zero"
                             "values:"
                             + str(np.unique(adjacency_matrix.data)))

        # Check if the diagonal is consistent (all 0s or  all 1s)
        if len(np.unique(adjacency_matrix.diagonal(0))) > 1:
            raise ValueError("The diagonal of the adjacency matrix must "
                             "be either all 1s or all 0s.")

        # If the diagonal is empty (all 0) fill it with 1s
        if adjacency_matrix.diagonal(0)[0] == 0:
            adjacency_matrix += sci_sparse_identity(adjacency_matrix.shape[0],
                                                    format='csr')
            # convert back to coo matrix (the addition is performed in CSR
            # format)
            adjacency_matrix = adjacency_matrix.tocoo()

        # Store the final checked version for serialization
        self.adjacency_mat = adjacency_matrix

        # Create an adjacency list from the adjacency matrix
        # Swap columns of the adjacency list to conform to the
        # _ragged_coo_graph_reduce [origin,destination] instead of the
        # [destination,origin] adjacency matrix required for ECM
        self.adjacency_list = tf.constant(np.array([adjacency_matrix.col,
                                                    adjacency_matrix.row]).transpose(),
                                          dtype=tf.int32)

        extremum = str(extremum).lower()
        if extremum in ['min', 'minimum']:
            self.extremum = Extremum.Min
        elif extremum in ['max', 'maximum']:
            self.extremum = Extremum.Max
        else:
            raise ValueError("Invalid 'extremum' argument.")

        if self.extremum == Extremum.Min:
            self._extremum_func = tf.reduce_min
        else:
            self._extremum_func = tf.reduce_max

    def call(self, inputs, *args, **kwargs):
        # Compute raw activations
        act = self.activation(inputs)
        # Perform pooling/reduction over the graph
        # Using this ragged implementation instead of tensor products results
        # in about 25x speedup on CPU, a massive speedup on GPU, and large
        # reduction in memory consumption
        ecm_act = _ragged_coo_graph_reduce(values=act,
                                        adjacency_list=self.adjacency_list,
                                        axis=-1,
                                        reduce_fn=self._extremum_func,
                                        sorted_adj_list=True  # The
                                        # adjacency list is sorted at build
                                        # time
                                        )
        # Bug Fix: This reshape is introduced to fix a weird bug at graph
        # creation time in evaluate and fit (keras). Code to reproduce the bug:
        # ecm_layer_max = layers.ExtremumConstraintModule(
        #     activation="linear",
        #     extremum="max",
        #     adjacency_matrix=adj_mat.transpose())
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
        return tf.reshape(tensor=ecm_act, shape=tf.shape(inputs))

    def build(self, input_shape: tf.TensorShape):
        # Sort the adjacency list according to the destination index
        # This enables to perform this operation only once instead of every
        # call to _ragged_coo_graph_reduce
        sorted_indices = tf.argsort(values=self.adjacency_list[:, 1],
                                    direction='ASCENDING',
                                    name='argsort_adj_list')
        self.adjacency_list = tf.gather(params=self.adjacency_list,
                                        indices=sorted_indices,
                                        axis=0,
                                        name='reorder_adj_list'
                                        )

    def get_config(self):
        config = {'extremum': str(self.extremum)}
        # FIXME reshape adj_mat to output it
        base_config = super(ExtremumConstraintModule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

        summed_l2 = tf.reduce_sum(tf.square(
            _dense_compute_hier_weight_diff_tensor(
                weights=concat_weights,
                adj_list=self.adj_list,
                axis=self.weights_vec_axis
            )
        ))
        self.add_loss(
            tf.cast(self.regularization_factor, dtype=summed_l2.dtype)
            * summed_l2)
        return super(DenseHierL2Reg, self).call(inputs=inputs)


@tf.function
def _ragged_coo_graph_reduce(values: tf.Tensor,
                             adjacency_list: tf.Tensor,
                             axis: int,
                             reduce_fn: callable,
                             sorted_adj_list=False,
                             ) -> tf.Tensor:
    """

    Parameters
    ----------
    values: a tf.Tensor containing the values.
    adjacency_list: adjacency list to be used to concatenate values in axis.
    The adjacency matrix is expected to be of shape [M,2], where M is the
    number of edges. The second is expected to denote [origin, destination].
    The destination of the edge denotes the index to be expanded while the
    origin the value to be expanded with.
    axis: axis on which graph expansion and reduction must be performed
    reduce_fn: reduction function, must support Ragged Tensors and comply with
    tf reduction ops arguments (input_tensor, axis, keepdims and name)
    sorted_adj_list: if False check and sort the adjacency list according to
    the destination index. Setting this parameter to True may avoid redundant
    operations. If set to True and the adjacency_list is not correctly sorted,
    will result in an error "Arguments to from_value_rowids do not form a
    valid RowPartition".

    Returns
    -------

    """
    # Preprocess inputs
    values = tf.convert_to_tensor(values)
    adjacency_list = tf.convert_to_tensor(adjacency_list, dtype=tf.int32)
    tf.assert_rank(adjacency_list, 2, message="The adjacency list must be a "
                                              "2D tensor (or tensor like "
                                              "object)")

    # Sort the adjacency list according to the destination index
    # This step is required for building a ragged Tensor from row_ids (row_ids
    # must be sorted in increasing order)
    if not sorted_adj_list:
        sorted_indices = tf.argsort(values=adjacency_list[:, 1],
                                    direction='ASCENDING',
                                    name='argsort_adj_list')
        adjacency_list = tf.gather(params=adjacency_list,
                                   indices=sorted_indices,
                                   axis=0,
                                   name='reorder_adj_list'
                                   )

    # Expand the values (makes a copy of the value denoted by the origin
    # index for every edge stemming from it)
    # I call this expansion assuming the adjacency list contains entries of
    # an adjacency matrix with a full diagonal. This is however not required,
    # and if the number of edges in the graph is less than the initial number
    # of values this operation will actually reduce the amounbt of values.
    # The conversion to a Ragged Tensor afterwards will restore the initial
    # dimension using empty Tensors. The result of the reduction operation will
    # depend on how the provided reducing function handles such empty tensors.
    expanded_values = tf.gather(params=values,
                                indices=adjacency_list[:, 0],
                                axis=axis,
                                name="get_neighbors_values")
    # Group according to the destination index using a Ragged Tensor
    # First swap axes to enable the use of ragged tensor construction
    # on first dimension
    expanded_values = keras_utils.utils.move_axis_to_first_dim(
        x=expanded_values,
        axis=axis)

    ragged_values = tf.RaggedTensor.from_value_rowids(
        values=expanded_values,
        value_rowids=adjacency_list[:, 1],
        nrows=tf.shape(values)[axis]
    )
    # Perform reduction
    # the 0 axis is the axis has been created above and has the size
    # of the original axis
    # the 1 axis is the ragged dimension of interest, to be reduced
    reduced_vals = reduce_fn(ragged_values, axis=1,
                             keepdims=False, name="ragged_reduction")

    # Now move the axis of interest back to its original place (0->axis)
    reduced_vals = keras_utils.utils.move_axis_to(input_tensor=reduced_vals,
                                                  axis_index=0,
                                                  new_index=axis)
    return reduced_vals


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
