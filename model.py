import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_model():

    model = keras.Sequential([
        
        layers.Input(shape=(28, 28, 1)),

        # First Convolution Block
        layers.Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second Convolution Block
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(pool_size=(2, 2)),

        # Fully Connected Layers
        layers.Flatten(),

        layers.Dense(128, activation="relu"),

        layers.Dropout(0.5),

        layers.Dense(10, activation="softmax")
    ])

    return model