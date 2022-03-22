from unittest import TestCase
from keras_utils import sparse
import tensorflow as tf
import numpy as np


# https://medium.com/@burnmg/software-testing-in-tensorflow-2-0-33c440ca908c
class Test(tf.test.TestCase):
    def test_expend_unit_dim(self):
        sp_t = tf.SparseTensor([[0, 0], [1, 1]],
                               tf.ones(2, dtype=tf.float32),
                               [2, 2])
        sp_t_exp = tf.sparse.expand_dims(sp_t, axis=0)
        sp_t_exp = sparse.expend_unit_dim(sp_t_exp,
                                          target_shape=tf.constant([2, 2, 2]))
        exp_out = np.array([[[1., 0.], [0., 1.]],
                            [[1., 0.], [0., 1.]]])
        self.assertIsInstance(sp_t_exp, tf.SparseTensor)
        self.assertShapeEqual(np.empty([2, 2, 2]),
                              tf.sparse.to_dense(sp_t_exp))
        self.assertAllEqual(exp_out, tf.sparse.to_dense(sp_t_exp))

    def test_sparse_dense_multiply(self):
        sp_t = tf.SparseTensor([[0, 0], [1, 1]], tf.ones(2, dtype=tf.float32),
                               [2, 2])
        sp_t_exp = tf.sparse.expand_dims(sp_t, axis=0)
        test = tf.ones(shape=[2, 2, 2], dtype=tf.float32)
        exp_out = np.array([[[1., 0.], [0., 1.]],
                            [[1., 0.], [0., 1.]]])
        exp_out2 = np.array([[[1., 0.], [0., 1.]],
                             [[2., 0.], [0., 2.]]])
        test2 = tf.concat([tf.ones([1, 2, 2]), tf.ones([1, 2, 2]) * 2], axis=0)
        for keep_sparse in [True, False]:
            mul = sparse.sparse_dense_multiply(sparse_t=sp_t_exp,
                                               dense_t=test,
                                               keep_sparse=keep_sparse)
            self.assertIsInstance(mul, tf.sparse.SparseTensor)
            self.assertShapeEqual(np.empty([2, 2, 2]),
                                  tf.sparse.to_dense(mul))
            self.assertAllEqual(exp_out, tf.sparse.to_dense(mul))
            mul2 = sparse.sparse_dense_multiply(sparse_t=sp_t_exp,
                                                dense_t=test2,
                                                keep_sparse=keep_sparse)
            self.assertIsInstance(mul2, tf.sparse.SparseTensor)
            self.assertShapeEqual(np.empty([2, 2, 2]),
                                  tf.sparse.to_dense(mul2))
            self.assertAllEqual(exp_out2, tf.sparse.to_dense(mul2))
