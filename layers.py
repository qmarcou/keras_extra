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

# Useful references:
# https://www.tensorflow.org/guide/keras/custom_layers_and_models

class Extremum (Enum):
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
            raise NotImplementedError("ECM computation using sparse operations"
                                      "is not yet implemented.")
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
                values=adjacency_matrix.data,
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
            self._extremum_func = tf.reduce_min
            self.extremum = Extremum.Min
        elif extremum in ['max', 'maximum']:
            self._extremum_func = tf.reduce_max
            self.extremum = Extremum.Max
        else:
            raise ValueError("Invalid 'extremum' argument.")

    def call(self, inputs, *args, **kwargs):
        # Compute raw activations
        act = self.activation(inputs)
        act = tf.reshape(act, shape=self._act_new_shape, name='act_reshape')

        # Compute the product of activation and adjacency mat
        # if self.sparse_adjacency:
        #     raise NotImplementedError("ECM computation using sparse operations"
        #                               "is not yet implemented.")
        # else:

        # Select values from the hierarchy and add them to the template filled
        # with 0 (if max) or np.inf (min)
        hier_act = tf.where(condition=self.adjacency_mat,
                            x=act,
                            y=self._filtered_act_template,
                            name="select_hier")

        extr_act = self._extremum_func(hier_act, axis=-1, keepdims=False,
                                       name="ecm_collapse")

        return extr_act

    def build(self, input_shape: tf.TensorShape):
        # reshape with length 1 first dimension for broadcasting over
        # sample and possibly other dimensions
        # input_shape does not contain the batch size dimension
        new_shape = tf.concat([tf.ones(shape=input_shape.rank - 1,
                                       dtype=tf.int32),
                               tf.fill(2, input_shape[-1])],
                              axis=0)
        # if this line throws an error it means the adjacency matrix does
        # not have the correct shape
        # if self.sparse_adjacency:
        #     self.adjacency_mat = tf.sparse.reshape(self.adjacency_mat,
        #                                            shape=new_shape)
        # else:
        self.adjacency_mat = tf.cast(self.adjacency_mat, dtype=tf.bool)
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

        # Create a base tensor to be filled with values
        act_template_shape = tf.ones(shape=input_shape.rank + 1,
                                     dtype=tf.int32)
        if self.extremum == Extremum.Min:
            self._filtered_act_template = tf.fill(value=np.Inf,
                                                  dims=act_template_shape)
        elif self.extremum == Extremum.Max:
            self._filtered_act_template = tf.zeros(shape=act_template_shape)

    def get_config(self):
        config = {'extremum': str(self.extremum),
                  'sparse': self.sparse_adjacency}
        base_config = super(ExtremumConstraintModule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

    def __init__(self, adjacency_matrix: np.ndarray | coo_matrix,
                 hier_side: str,
                 sparse_adjacency: bool = False,
                 **kwargs):
        super(DenseHierL2Reg, self).__init__(**kwargs)
        self.sparse_adjacency = sparse_adjacency
        self.sparse_adjacency = adjacency_matrix

        hier_side = str(hier_side).lower()
        if hier_side in ['in', 'input']:
            self.hier_side = "input"
        elif hier_side in ['out', 'output']:
            self.hier_side = "output"
        else:
            raise ValueError("Invalid 'extremum' argument.")

    def call(self, inputs):
        # call a tf.func computing the L2 norm of difference with each parent
        self.add_loss(self.weights, self.bias, self.hierarchy)
