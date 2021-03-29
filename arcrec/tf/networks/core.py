from typing import Dict

import tensorflow as tf

from ...features import *
from ..utils import create_embedding_dict, get_feature_tensors


class DNN(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_units,
        use_bias=True,
        use_bn=False,
        dropout=0,
        activation="relu",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        bias_regularizer=None,
        **kwargs
    ):
        super(DNN, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.activation = activation
        self.dropout = dropout

        self.dense_layers = [
            tf.keras.layers.Dense(
                units=hidden_units[i],
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                use_bias=use_bias,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
            )
            for i in range(len(hidden_units))
        ]

        self.activation_layers = [
            tf.keras.layers.Activation(self.activation)
            for _ in range(len(hidden_units))
        ]

        if self.dropout > 0:
            self.dropout_layers = [
                tf.keras.layers.Dropout(self.dropout)
                for _ in range(len(hidden_units))
            ]

        if self.use_bn:
            self.bn_layers = [
                tf.keras.layers.BatchNormalization()
                for _ in range(len(self.hidden_units))
            ]

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for i in range(len(self.hidden_units)):
            x = self.dense_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x, training=training)
            x = self.activation_layers[i](x)
            if self.dropout > 0:
                x = self.dropout_layers[i](x, training=training)
        return x


class Linear(tf.keras.layers.Layer):
    def __init__(
        self,
        feature_columns,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    ):
        super(Linear, self).__init__()
        self.feature_columns = feature_columns
        sparse_feature_columns = [
            feat._replace(embedding_dim=1)
            for feat in (
                get_sparse_feature_columns(feature_columns)
                + get_seq_sparse_feature_columns(feature_columns)
            )
        ]
        self.sparse_embedding_dict = create_embedding_dict(
            sparse_feature_columns,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
        )

        self.dense_feature_columns = get_dense_feature_columns(feature_columns)
        if len(self.dense_feature_columns) > 0:
            self.dense_kernel = tf.keras.layers.Dense(
                units=1,
                activation=None,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )
        self.use_bias = use_bias
        if use_bias:
            self.bias = tf.Variable(0.0, name="linear_bias")

    def call(self, features: Dict[str, tf.Tensor]):
        sparse_tensors, dense_tensors = get_feature_tensors(
            self.feature_columns, features, self.sparse_embedding_dict
        )
        logits = []
        if len(sparse_tensors) > 0:
            sparse_tensors = tf.concat(sparse_tensors, axis=1)
            sparse_logit = tf.reduce_sum(
                sparse_tensors, axis=1, keepdims=False
            )
            logits.append(sparse_logit)
        if len(dense_tensors) > 0:
            dense_tensors = tf.concat(dense_tensors, axis=-1)
            dense_logit = self.dense_kernel(dense_tensors)
            logits.append(dense_logit)
        logits = tf.add_n(logits)
        if self.use_bias:
            logits += self.bias
        return logits
