import sys

import tensorflow as tf

sys.path.append("../")
from arcrec.tf.models import WideDeepModel

from criteo_utils import create_criteo_dataset


if __name__ == "__main__":
    file = "../train.txt"
    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embedding_dim = 8
    dnn_dropout = 0.5
    dnn_hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    (
        (linear_feature_columns, dnn_feature_columns),
        (train_X, train_y),
        (test_X, test_y),
    ) = create_criteo_dataset(
        file=file,
        embedding_dim=embedding_dim,
        read_part=read_part,
        sample_num=sample_num,
        test_size=test_size,
    )
    model = WideDeepModel(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        dnn_hidden_units=dnn_hidden_units,
    )
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.AUC()],
    )
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=1, restore_best_weights=True
            )
        ],
        batch_size=batch_size,
        validation_split=0.1,
    )
    print(
        "test AUC: %f"
        % model.evaluate(test_X, test_y, batch_size=batch_size)[1]
    )
    model.save("./my_model", save_format="tf")
