import keras
import tensorflow as tf
from keras import Sequential, layers, models
from models.lib import get_augmentation_layer

def generate_model(input_shape=(256, 256, 3), num_classes=45, growth_rate=32):
    def dense_block(x, blocks):
        for _ in range(blocks):
            shortcut = x
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(4 * growth_rate, (1, 1), padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(growth_rate, (3, 3), padding="same", use_bias=False)(x)
            x = layers.concatenate([shortcut, x])
        return x

    def transition_layer(x, reduction=0.5):
        filters = int(x.shape[-1] * reduction)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
        x = layers.AveragePooling2D((2, 2), strides=2)(x)
        return x

    inputs = layers.Input(shape=input_shape)
    x = get_augmentation_layer()(inputs)

    x = layers.Conv2D(64, (7, 7), strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    x = dense_block(x, blocks=4)
    x = transition_layer(x)
    x = dense_block(x, blocks=4)
    x = transition_layer(x)
    x = dense_block(x, blocks=4)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="DenseNetMini_Scratch")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model
