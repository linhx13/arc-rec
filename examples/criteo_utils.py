import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.append("../")
from arcrec.features import *


def create_criteo_dataset(
    file, embedding_dim=8, read_part=True, sample_num=100000, test_size=0.2
):
    """
    a example about creating criteo dataset
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    names = [
        "label",
        "I1",
        "I2",
        "I3",
        "I4",
        "I5",
        "I6",
        "I7",
        "I8",
        "I9",
        "I10",
        "I11",
        "I12",
        "I13",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
    ]

    if read_part:
        data = pd.read_csv(
            file, sep="\t", iterator=True, header=None, names=names
        )
        data = data.get_chunk(sample_num)

    else:
        data = pd.read_csv(file, sep="\t", header=None, names=names)

    sparse_features = ["C" + str(i) for i in range(1, 27)]
    dense_features = ["I" + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna("-1")
    data[dense_features] = data[dense_features].fillna(0)
    target = ["label"]

    for feat in sparse_features:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_columns = [
        SparseFeature(
            feat, vocab_size=data[feat].max() + 1, embedding_dim=embedding_dim
        )
        for feat in sparse_features
    ]
    dense_feature_columns = [DenseFeature(feat, 1) for feat in dense_features]

    feature_columns = sparse_feature_columns + dense_feature_columns

    linear_feature_columns = feature_columns
    dnn_feature_columns = feature_columns

    # linear_feature_columns = sparse_feature_columns
    # dnn_feature_columns = dense_feature_columns

    train, test = train_test_split(data, test_size=test_size)

    train_X = {feat.name: train[feat.name].values for feat in feature_columns}
    train_y = train["label"].values.astype("int32")

    test_X = {feat.name: test[feat.name].values for feat in feature_columns}
    test_y = test["label"].values.astype("int32")

    return (
        (linear_feature_columns, dnn_feature_columns),
        (train_X, train_y),
        (test_X, test_y),
    )
