"""A collection of custom keras layers."""
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
                 adjacency_matrix: coo_matrix,
                 **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = False
        self.activation = activations.get(activation)

        if isspmatrix(adjacency_matrix):
            if not isspmatrix_coo(adjacency_matrix):
                adjacency_matrix = adjacency_matrix.tocoo()
        else:
            adjacency_matrix = coo_matrix(adjacency_matrix)
        # Check that provided matrix is square
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        # Create a sparse Tensor from the hierarchy information
        self.adjacency_mat = tf.SparseTensor(
            indices=np.mat([adjacency_matrix.row,
                            adjacency_matrix.col])
                .transpose(),
            values=adjacency_matrix.data,
            dense_shape=adjacency_matrix.shape)

        # reorder in row major as advised in tf docs
        self.adjacency_mat = tf.sparse.reorder(self.adjacency_mat)

        extremum = str(extremum)
        if extremum in ['min', 'minimum']:
            self.extremum_func = tf.reduce_min
            self.extremum_str = "min"
        elif extremum in ['max', 'maximum']:
            self.extremum_func = tf.reduce_max
            self.extremum_str = "max"
        else:
            raise ValueError("Invalid 'extremum' argument.")

    def call(self, inputs):
        # Compute raw activations
        act = self.activation(inputs)
        act_shape = act.shape
        # Add a broadcasting dimension
        new_shape = tf.concat([act_shape[:-1],
                               tf.ones(shape=1, dtype=tf.int32),
                               act_shape[-1:]])
        act = inputs.reshape(act, shape=new_shape)

        # Adjust adjacency_mat shape to use broadcast
        if tf.rank(self.adjacency_mat) != tf.rank(act):
            # reshape with length 1 first dimension for broadcasting over
            # sample and possibly other dimensions
            # rank should be known for both even at compile time(?)
            new_shape = tf.concat([tf.ones(shape=act_shape[:-1].shape,
                                           dtype=tf.int32),
                                   tf.constant([act_shape[-1]*2],
                                               dtype=tf.int32)])
            # if this line throws an error it means the adjacency matrix does
            # not have the correct shape
            self.adjacency_mat = tf.sparse.reshape(self.adjacency_mat,
                                                   shape=new_shape)




        return

    def get_config(self):
        config = {'extremum': self.extremum_str}
        base_config = super(ExtremumConstraintModule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
