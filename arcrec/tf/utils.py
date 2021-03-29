from typing import Dict

import tensorflow as tf

from ..features import *


def create_embedding_dict(
    feature_columns,
    embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
    seq_mask_zero=True,
):
    sparse_feature_columns = get_sparse_feature_columns(feature_columns)
    seq_sparse_feature_columns = get_seq_sparse_feature_columns(
        feature_columns
    )
    embedding_dict = {}
    for feat in sparse_feature_columns:
        embedding = tf.keras.layers.Embedding(
            input_dim=feat.vocab_size,
            output_dim=feat.embedding_dim,
            embeddings_initializer=feat.embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            name=feat.name + "_embedding",
        )
        embedding_dict[feat.name] = embedding
    for feat in seq_sparse_feature_columns:
        embedding = tf.keras.layers.Embedding(
            input_dim=feat.vocab_size,
            output_dim=feat.embedding_dim,
            embeddings_initializer=feat.embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            mask_zero=seq_mask_zero,
            name=feat.name + "_seq_embedding",
        )
        embedding_dict[feat.name] = embedding
    return embedding_dict


def embedding_lookup(
    feature_columns, features: Dict[str, tf.Tensor], embedding_dict
):
    tensors = [
        embedding_dict[feat.name](features[feat.name])
        for feat in feature_columns
    ]
    return tensors


def seq_embedding_lookup(
    feature_columns, features: Dict[str, tf.Tensor], embedding_dict
):
    pass


def get_feature_tensors(
    feature_columns,
    features: Dict[str, tf.Tensor],
    embedding_dict,
    ret_feature_columns=False,
):
    sparse_features = []
    sparse_feature_columns = get_sparse_feature_columns(feature_columns)
    sparse_feature_tensors = embedding_lookup(
        sparse_feature_columns, features, embedding_dict
    )
    # TODO: seq_sparse_feature_tensors
    dense_feature_columns = get_dense_feature_columns(feature_columns)
    dense_feature_tensors = [
        features[feat.name] for feat in dense_feature_columns
    ]
    res = (sparse_feature_tensors, dense_feature_tensors)
    if ret_feature_columns:
        res = res + (sparse_feature_columns, dense_feature_columns)
    return res


def combine_dnn_tensors(sparse_tensors, dense_tensors):
    tensors = []
    if len(sparse_tensors) > 0:
        tensors.append(
            tf.keras.layers.Flatten()(tf.concat(sparse_tensors, axis=-1))
        )
    if len(dense_tensors) > 0:
        tensors.append(
            tf.keras.layers.Flatten()(tf.concat(dense_tensors, axis=-1))
        )
    return tf.concat(tensors, axis=-1)


def group_embedding_by_dim(embedding_dict: Dict[str, tf.Tensor]):
    groups = dict()
    for embedding in embedding_dict.values():
        dim = embedding.shape[-1]
        if dim not in groups:
            groups[dim] = [embedding]
        else:
            groups[dim].append(embedding)
    return groups
