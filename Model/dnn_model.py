from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input
)

from tensorflow.keras.optimizers import Adam


def build_dnn_model(input_shape):

    model = Sequential()

    # Input Layer
    model.add(Input(shape=(input_shape,)))

    # Hidden Layer 1
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.40))

    # Hidden Layer 2
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))

    # Hidden Layer 3
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    # Hidden Layer 4
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Hidden Layer 5
    model.add(Dense(32, activation="relu"))

    # Output Layer
    model.add(Dense(5, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_dnn_model(20480)

    model.summary()