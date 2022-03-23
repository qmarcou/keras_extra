"""A collection of custom keras layers."""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix, isspmatrix_coo, isspmatrix
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation
from keras import activations
from tensorflow.keras import backend as K
from typing import Optional


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
        super(Activation, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = False
        self.activation = activations.get(activation)
        self.sparse_adjacency = sparse_adjacency

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
                                                   num_rows=adjacency_matrix.shape[0],
                                               num_columns=adjacency_matrix.shape[0],
                                               dtype=self.adjacency_mat.dtype))
        else:
            self.adjacency_mat = tf.constant(adjacency_matrix)
            # Add ones on the diagonal to include considered class predictions
            self.adjacency_mat = tf.add(self.adjacency_mat,
                                        tf.eye(num_rows=adjacency_matrix
                                               .shape[0],
                                               num_columns=adjacency_matrix
                                               .shape[0],
                                               dtype=self.adjacency_mat.dtype))

        extremum = str(extremum)
        if extremum in ['min', 'minimum']:
            self.extremum_func = [tf.divide, tf.reduce_min]
            self.extremum_str = "min"
        elif extremum in ['max', 'maximum']:
            self.extremum_func = [tf.multiply, tf.reduce_max]
            self.extremum_str = "max"
        else:
            raise ValueError("Invalid 'extremum' argument.")

    def call(self, inputs):
        # Compute raw activations
        act = self.activation(inputs)
        act_shape = tf.shape(act)
        # Add a broadcasting dimension to the activation tensor
        new_shape = tf.concat([act_shape[:-1],
                               tf.ones(shape=1, dtype=tf.int32),
                               act_shape[-1:]],
                              axis=0)
        act = tf.reshape(act, shape=new_shape)

        # Cast adjacency mat to the correct dtype
        # This operation should only happen once
        if self.adjacency_mat.dtype != act.dtype:
            self.adjacency_mat = tf.cast(self.adjacency_mat, act.dtype)

        # Adjust adjacency_mat shape to use broadcast if needed
        # This operation should only happen once
        if tf.rank(self.adjacency_mat) != tf.rank(act):
            # reshape with length 1 first dimension for broadcasting over
            # sample and possibly other dimensions
            # rank should be known for both even at compile time(?)
            new_shape = tf.concat([tf.ones(shape=tf.shape(act_shape[:-1]),
                                           dtype=tf.int32),
                                   tf.fill(2, act_shape[-1])],
                                  axis=0)
            # if this line throws an error it means the adjacency matrix does
            # not have the correct shape
            if self.sparse_adjacency:
                self.adjacency_mat = tf.sparse.reshape(self.adjacency_mat,
                                                       shape=new_shape)
            else:
                self.adjacency_mat = tf.reshape(self.adjacency_mat,
                                                shape=new_shape)

        # Compute the product of activation and adjacency mat
        if self.sparse_adjacency:
            raise NotImplementedError("ECM computation using sparse operations"
                                      "is not yet implemented.")
        else:
            # multiply by 0/1 to select predictions from parents of a class if
            # max, and divide to cast values ot infinity if min
            hier_act = self.extremum_func[0](act, self.adjacency_mat)
            extr_act = self.extremum_func[1](hier_act, axis=-1, keepdims=False,
                                             name="ECM collapse")

        return extr_act

    def get_config(self):
        config = {'extremum': self.extremum_str,
                  'sparse': self.sparse_adjacency}
        base_config = super(ExtremumConstraintModule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
