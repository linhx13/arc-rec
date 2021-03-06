from typing import Dict

import tensorflow as tf

from ..networks import Linear, DNN
from ..utils import (
    create_embedding_dict,
    get_feature_tensors,
    combine_dnn_tensors,
)


class WideDeepModel(tf.keras.Model):
    def __init__(
        self,
        linear_feature_columns,
        dnn_feature_columns,
        embedding_regularizer=tf.keras.regularizers.l2(1e-5),
        linear_use_bias=False,
        linear_kernel_initializer="glorot_uniform",
        linear_kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        dnn_hidden_units=(128, 128),
        dnn_use_bias=False,
        dnn_use_bn=False,
        dnn_dropout=0,
        dnn_activation="relu",
        dnn_kernel_initializer="glorot_uniform",
        dnn_bias_initializer="zeros",
        dnn_kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        dnn_bias_regularizer=None,
    ):
        super(WideDeepModel, self).__init__()

        self.linear_model = Linear(
            linear_feature_columns,
            linear_use_bias,
            linear_kernel_initializer,
            linear_kernel_regularizer,
        )
        self.dnn_model = DNN(
            dnn_hidden_units,
            dnn_use_bias,
            dnn_use_bn,
            dnn_dropout,
            dnn_activation,
            dnn_kernel_initializer,
            dnn_bias_initializer,
            dnn_kernel_regularizer,
            dnn_bias_regularizer,
        )
        self.dnn_dense = tf.keras.layers.Dense(
            1,
            use_bias=False,
            kernel_initializer=dnn_kernel_initializer,
            kernel_regularizer=dnn_kernel_regularizer,
            bias_initializer=dnn_bias_initializer,
            bias_regularizer=dnn_bias_regularizer,
        )
        self.dnn_feature_columns = dnn_feature_columns
        self.sparse_embedding_dict = create_embedding_dict(dnn_feature_columns)
        self.final = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, features: Dict[str, tf.Tensor], **kwargs):
        linear_logit = self.linear_model(features)
        sparse_feature_tensors, dense_feature_tensors = get_feature_tensors(
            self.dnn_feature_columns, features, self.sparse_embedding_dict
        )
        dnn_input = combine_dnn_tensors(
            sparse_feature_tensors, dense_feature_tensors
        )
        dnn_output = self.dnn_model(dnn_input)
        dnn_logit = self.dnn_dense(dnn_output)

        output = self.final(tf.add_n([linear_logit, dnn_logit]))
        return output
