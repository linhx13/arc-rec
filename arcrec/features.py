from collections import namedtuple

from tensorflow.keras.initializers import RandomNormal, Zeros


class DenseFeature(namedtuple("DenseFeature", ["name", "dim", "dtype"])):

    __slots__ = ()

    def __new__(cls, name, dim=1, dtype="float32"):
        return super(DenseFeature, cls).__new__(cls, name, dim, dtype)


class SparseFeature(
    namedtuple(
        "SparseFeature",
        [
            "name",
            "vocab_size",
            "embedding_dim",
            "use_hash",
            "dtype",
            "embeddings_initializer",
        ],
    )
):
    __slots__ = ()

    def __new__(
        cls,
        name,
        vocab_size,
        embedding_dim=4,
        use_hash=False,
        dtype="int32",
        embeddings_initializer=None,
    ):
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocab_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(
                mean=0.0, stddev=0.0001, seed=42
            )
        return super(SparseFeature, cls).__new__(
            cls,
            name,
            vocab_size,
            embedding_dim,
            use_hash,
            dtype,
            embeddings_initializer,
        )

    def __hash__(self):
        return self.name.__hash__()


class SeqSparseFeature(
    namedtuple(
        "SeqSparseFeature",
        [
            "name",
            "vocab_size",
            "max_length",
            "embedding_dim",
            "use_hash",
            "dtype",
            "embeddings_initializer",
            "combiner",
        ],
    )
):

    __slots__ = ()

    def __new__(
        cls,
        name,
        vocab_size,
        max_length,
        embedding_dim=4,
        use_hash=False,
        dtype="int32",
        embeddings_initializer=None,
        combiner="mean",
    ):
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocab_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(
                mean=0.0, stddev=0.0001, seed=42
            )
        return super(SeqSparseFeature, cls).__new__(
            cls,
            name,
            vocab_size,
            max_length,
            embedding_dim,
            use_hash,
            dtype,
            embeddings_initializer,
            combiner,
        )


def get_dense_feature_columns(feature_columns):
    return [feat for feat in feature_columns if isinstance(feat, DenseFeature)]


def get_sparse_feature_columns(feature_columns):
    return [
        feat for feat in feature_columns if isinstance(feat, SparseFeature)
    ]


def get_seq_sparse_feature_columns(feature_columns):
    return [
        feat for feat in feature_columns if isinstance(feat, SeqSparseFeature)
    ]
